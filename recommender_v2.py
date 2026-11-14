"""recommender_v2.py · recomendador de producción (team_40) · versión 3: "la práctica es un laboratorio".

    python recommender_v2.py             # producción
    python recommender_v2.py --dry-run   # calcula respuestas pero NO las inserta

Idea central
------------
La ronda de práctica NO puntúa; solo cuentan las 2 horas finales. Así que hasta el domingo a las
10:00 el sistema no intenta maximizar ventas: hace EXPERIMENTOS contra el simulador real para
aprender cómo decide, y en la final explota lo aprendido.

  1. A/B en vivo de dos formas de elegir candidatos (50/50 por hash de la petición):
       · "full": el modelo puntúa TODO el catálogo disponible (~5.000 productos, ~40 ms).
       · "cand": retrieval clásico (~60-90 candidatos) + re-ranking.
     La conversión real de cada brazo sale de team_NN.feedback y se guarda en state/live_stats.json.
     Al empezar la final se elige solo el brazo ganador.
  2. Exploración diseñada (EXPLORE_RATE de las peticiones, mitad y mitad):
       · "+shuf": el top-10 del modelo en orden aleatorio → mide el efecto de la posición sin
         confundirlo con la relevancia.
       · "+wide": 10 productos muestreados del top-150 (más uno al azar del catálogo), en orden
         aleatorio → etiquetas del simulador real sobre productos que nadie más está enseñando.
  3. train_model.py --loop reentrena cada hora con ese feedback (que es el modelo oculto, no el
     log del sistema antiguo) y este proceso recarga el modelo en caliente.
  4. Desde FINAL_START_UTC (domingo 10:00 CET): sin exploración, sin grupo de control, brazo ganador.

Degradación: modelo LightGBM → best sellers en stock (= baseline). Nunca se responde algo inválido.

Por defecto va a por el marcador: exploración y grupo de control apagados (se reactivan con
EXPLORE_RATE=0.3 y HOLDOUT=0.02). El A/B entre "full" y "cand" sigue activo porque no cuesta conversión.

El `model_version` que aparece en el marcador es siempre el mismo (PUBLIC_NAME, por defecto "v4_k4"):
no revela la estrategia. Los brazos y modos se siguen distinguiendo en las estadísticas internas.

Variables de entorno: EXPLORE_RATE (0), HOLDOUT (0), PUBLIC_NAME, ARM (ab | full | cand; por defecto ab en
práctica y automático en la final), REBUY_PENALTY (1.0 = decide el modelo), FINAL_START_UTC,
MODEL_PATH, CACHE_DIR, POLL_SECONDS (0.25).
"""
from __future__ import annotations

import json
import os
import pickle
import random
import subprocess
import sys
import threading
import time
import zlib
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import features_cache as fc
from features_cache import F, FeatureCache, log

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE / "models" / "ranker.pkl"))
STATE_PATH = Path(os.environ.get("STATE_PATH", HERE / "state" / "live_stats_v7.json"))
POLL = float(os.environ.get("POLL_SECONDS", 0.15))
MAX_AGE = 6.0            # peticiones más viejas que esto ya han recibido el fallback: no se contestan
EXPLORE_RATE = float(os.environ.get("EXPLORE_RATE", 0.0))
HOLDOUT = float(os.environ.get("HOLDOUT", 0.0))
PUBLIC_NAME = os.environ.get("PUBLIC_NAME", "v7")   # prefijo del model_version que aparece en el marcador
LABELS_PATH = STATE_PATH.with_name("labels.json")
# Brazos: cand = ~80 candidatos · wide = ~250 candidatos (aprovecha el margen de tiempo) · full = todo el catálogo.
# ab (por defecto) reparte 50/50 entre cand y wide; en la final se queda solo el que más compras reales consiga.
ARM = os.environ.get("ARM", "ab").lower()
SLOW_COMPUTE_MS = float(os.environ.get("SLOW_COMPUTE_MS", 150))   # por encima, el catálogo completo se apaga solo
SLOW_P95_MS = float(os.environ.get("SLOW_P95_MS", 2400))
# Regla del simulador (aclarada por la organización): quien ya compró un producto casi nunca lo vuelve a
# comprar cuando se lo enseñamos. El histórico se generó SIN ese efecto, así que el modelo aprendido de él
# tiende a recomendar justo lo ya comprado. Aquí se manda al fondo de la lista todo lo que el usuario
# compró (histórico + eventos en vivo + nuestras propias recomendaciones).
REBUY_PENALTY = float(os.environ.get("REBUY_PENALTY", 0.02))
FINAL_START = fc.parse_ts(os.environ.get("FINAL_START_UTC", "2026-11-15T09:00:00Z")) or 0.0
MAX_FULL_BATCH = 5       # con más peticiones en el lote se usa "cand" para proteger la latencia
WIDE_POOL = 150
PRED_THREADS = max(1, min(4, (os.cpu_count() or 2) - 1))
AUTO_TRAIN = os.environ.get("AUTO_TRAIN", "1") != "0"      # lanza y vigila train_model.py --loop en otro proceso
INPROC_BUILD = os.environ.get("INPROC_BUILD", "0") == "1"  # solo para pruebas
TRAIN_EVERY_MIN = int(os.environ.get("TRAIN_EVERY_MIN", 60))

SQL_BEST_SELLERS = """
SELECT product_id, count() AS n
FROM shop.history_events
WHERE event_type = 'purchase'
GROUP BY product_id"""

SQL_NEW_REQUESTS = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 60"""
SQL_NEW_REQUESTS_ALT = "SELECT * FROM shop.requests ORDER BY ts DESC LIMIT 60"
# lectura en ventana corta: no ordena toda la tabla, así que cada vuelta del bucle es más rápida
SQL_NEW_REQUESTS_FAST = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
WHERE ts > now() - INTERVAL 8 SECOND
ORDER BY ts DESC
LIMIT 60"""

SQL_FEEDBACK_RECENT = f"""
SELECT request_id, ts, shown, served_by, fallback_reason, clicked, purchased, latency_ms
FROM {fc.TEAM}.feedback
WHERE ts > now() - INTERVAL 3 MINUTE AND request_id != '__init__'"""


# ----------------------------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------------------------
class Ranker:
    """Modelo LightGBM entrenado por train_model.py."""

    def __init__(self, bundle: dict) -> None:
        import lightgbm as lgb

        if bundle.get("features") != fc.FEATURES:
            raise RuntimeError("el modelo se entrenó con otra lista de features: vuelve a entrenar")
        self.combine = str(bundle.get("combine", "click"))
        self.click = lgb.Booster(model_str=bundle["click"])
        self.buy = lgb.Booster(model_str=bundle["buy"]) if bundle.get("buy") and self.combine in ("click_x_buy", "blend") else None
        self.direct = lgb.Booster(model_str=bundle["direct"]) if bundle.get("direct") and self.combine in ("direct", "blend") else None
        self.vocab = bundle.get("vocab", {})
        self.version = str(bundle.get("version", "lgb"))
        self.wide_default = bool(bundle.get("wide_pool", False))   # lo que dijo la validación offline
        self.cache_blocks = set(bundle.get("cache_blocks", [])) - {"vistas diarias"}

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "Ranker":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def score(self, X: np.ndarray) -> np.ndarray:
        """P(compra) según la combinación que ganó en validación (train_model.py la elige)."""
        th = PRED_THREADS if len(X) > 2000 else 1
        if self.combine == "direct" and self.direct is not None:
            p = self.direct.predict(X, num_threads=th)
        else:
            p = self.click.predict(X, num_threads=th)
            if self.buy is not None:
                p = p * self.buy.predict(X, num_threads=th)
            if self.combine == "blend" and self.direct is not None:
                p = np.sqrt(np.maximum(p, 1e-12) * np.maximum(self.direct.predict(X, num_threads=th), 1e-12))
        return np.nan_to_num(np.asarray(p, np.float64), nan=0.0)


def rank_batch(cache: FeatureCache, ranker: Ranker | None, recs: list[dict], arms: list[str] | None = None,
               explore: list[str | None] | None = None, rng: random.Random | None = None,
               ) -> list[tuple[list[str] | None, str, object]]:
    """Ranking de un lote → [(10 product_ids | None, etiqueta de versión, Request | None)].

    arms[i]    : "full" (todo el catálogo) o "cand" (retrieval + re-ranking).
    explore[i] : None (explotar), "shuf" (top-10 barajado) o "wide" (muestreo del top-150 barajado).
    Sin modelo devuelve None: el llamante responde best sellers.
    """
    n = len(recs)
    arms = arms or ["cand"] * n
    explore = explore or [None] * n
    if ranker is None:
        return [(None, "v3-bestsellers", None)] * n
    prepared: list[tuple | None] = []
    full_pool = None
    for rec, arm in zip(recs, arms):
        try:
            R = cache.parse_request(rec)
            S = cache.session(R)
            if arm == "full":
                if full_pool is None:
                    full_pool = cache.all_candidates()
                cands = full_pool
            else:
                cands = cache.candidates(R, S, wide=arm == "wide")
            prepared.append((R, cands, cache.featurize(R, S, cands, ranker.vocab)))
        except Exception as e:  # noqa: BLE001  una petición rara no tumba el lote
            log(f"petición {rec.get('request_id')} no procesable: {str(e)[:150]}")
            prepared.append(None)
    ok = [p for p in prepared if p is not None]
    scores = ranker.score(np.vstack([p[2] for p in ok])) if ok else None
    out: list[tuple[list[str] | None, str, object]] = []
    off = 0
    pids = cache.pids
    for p, arm, exp in zip(prepared, arms, explore):
        if p is None:
            out.append((None, "error", None))
            continue
        R, cands, X = p
        s = np.array(scores[off: off + len(cands)], np.float64)
        off += len(cands)
        if REBUY_PENALTY != 1.0:
            s[X[:, F["user_bought"]] > 0] *= REBUY_PENALTY
        k = min(len(s), WIDE_POOL if exp == "wide" else 10)
        part = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        order = part[np.argsort(-s[part], kind="stable")]
        label = f"v3-{ranker.version}-{arm}"
        if exp == "shuf" and rng is not None:
            top = cands[order[:10]].tolist()
            rng.shuffle(top)
            label += "+shuf"
        elif exp == "wide" and rng is not None and len(order) > 12:
            w = 1.0 / (np.arange(len(order)) + 5.0) ** 0.8          # más peso a lo mejor puntuado
            pick = np.random.default_rng(rng.getrandbits(32)).choice(len(order), size=9, replace=False, p=w / w.sum())
            top = cands[order[pick]].tolist()
            pool = full_pool if full_pool is not None else cache.all_candidates()
            for _ in range(20):                                     # + 1 producto al azar del catálogo
                extra = int(pool[rng.randrange(len(pool))])
                if extra not in top:
                    top.append(extra)
                    break
            rng.shuffle(top)
            label += "+wide"
        else:
            top = cands[order[:10]].tolist()
        out.append(([pids[i] for i in top], label, R))
    return out


# ----------------------------------------------------------------------------------------------
# Motor de producción
# ----------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.best: list[str] = []
        self.unavailable: set[str] = set()
        self.cache: FeatureCache | None = None
        self.ranker: Ranker | None = None
        self._model_mtime = 0.0
        self.rng = random.Random()
        self.stats: Counter = Counter()
        self.ages: list[float] = []
        self.compute_ms: list[float] = []
        self.req_meta: OrderedDict[str, tuple[str, str, bool]] = OrderedDict()  # request_id → (user, etiqueta, cold)
        self.fb_seen: OrderedDict[str, int] = OrderedDict()
        self.fb_modes: dict[str, list[int]] = {}   # etiqueta → [peticiones, clics, compras]
        self.fb_window: Counter = Counter()
        self.final_arm: str | None = None
        self.public: dict[str, str] = {}
        self._last_report = time.time()
        self._req_sql = 0
        self._pull_fail = 0
        self._empty = 0
        self._warned = ""
        self.trainer = None
        self._trainer_started = 0.0
        self.started = time.time()
        self.req_items: OrderedDict[str, tuple] = OrderedDict()   # lo que enviamos, para contrastar con el feedback
        self.srv_latency: list[float] = []
        self._last_loop = time.time()
        self.pushers = ThreadPoolExecutor(max_workers=3, thread_name_prefix="push")
        self.slow_until = 0.0
        try:
            self.fb_modes = {k: list(v) for k, v in json.loads(STATE_PATH.read_text()).items()}
            log(f"estadísticas en vivo recuperadas de {STATE_PATH.name} ({len(self.fb_modes)} etiquetas)")
        except Exception:  # noqa: BLE001
            pass

    # --- hilos en segundo plano -------------------------------------------------------------
    def _spawn(self, name: str, fn, period: float) -> None:
        def loop() -> None:
            fails = 0
            while True:
                t = time.time()
                try:
                    fn()
                    if fails:
                        log(f"[{name}] recuperado")
                    fails = 0
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    if fails in (1, 10) or fails % 100 == 0:
                        log(f"[{name}] error ({fails} seguidos): {str(e)[:200]}")
                time.sleep(max(0.05, period - (time.time() - t)))

        threading.Thread(target=loop, name=name, daemon=True).start()

    def _refresh_stock(self) -> None:
        pids = {str(r["product_id"]) for r in fc.query(fc.SQL_OUT_OF_STOCK, timeout=4, name="out_of_stock")}
        self.unavailable = pids
        cache = self.cache
        if cache is not None:
            cache.set_oos_fast(pids)

    def _watch_model(self) -> None:
        if not MODEL_PATH.exists():
            return
        mtime = MODEL_PATH.stat().st_mtime
        if mtime == self._model_mtime:
            return
        self._model_mtime = mtime          # si el fichero no sirve no se reintenta hasta que cambie
        ranker = Ranker.load(MODEL_PATH)
        ranker.score(np.zeros((2, fc.NF), np.float32))  # prueba de humo antes de publicarlo
        self.ranker = ranker
        log(f"modelo cargado: {ranker.version} (puntuación: {ranker.combine}; "
            f"validación offline prefiere {'wide' if ranker.wide_default else 'cand'})")

    def consistency_warning(self) -> str:
        """El modelo solo vale si la caché en servicio tiene los MISMOS bloques con los que se entrenó.
        Servir con una caché parcial (sin co-views, sin compras por usuario…) deja esas features a cero
        y hunde la conversión sin dar ningún error."""
        cache, ranker = self.cache, self.ranker
        if cache is None:
            return ""
        bad = sorted(k for k, v in cache.status.items() if v != "OK")
        missing = sorted(ranker.cache_blocks - {k for k, v in cache.status.items() if v == "OK"}) if ranker else []
        if bad or missing:
            return (f"¡¡ CACHÉ INCOMPLETA !! bloques fallidos: {bad or '-'} · bloques que el modelo usa y faltan: "
                    f"{missing or '-'} → para, ejecuta `python features_cache.py --build` y vuelve a arrancar")
        return ""

    def _init_cache(self) -> None:
        while self.cache is None:
            try:
                cache = None
                if fc.LIVE_CACHE_PATH.exists():
                    try:
                        cache = FeatureCache.load(fc.LIVE_CACHE_PATH)
                        cache._upd_ts = 0.0               # fuerza una lectura completa de catalog_updates
                        cache._ev_ts, cache.user_ev = 0.0, {}
                        log(f"caché cargada de disco ({cache.n} productos)")
                    except Exception as e:  # noqa: BLE001
                        log(f"caché de disco no válida ({str(e)[:120]}): la reconstruyo")
                if cache is None and INPROC_BUILD:
                    cache = FeatureCache.build()
                    cache.save(fc.LIVE_CACHE_PATH)
                elif cache is None:
                    # En OTRO proceso: construirla aquí dentro frenaría el bucle que responde peticiones.
                    log("no hay caché: la construyo en un proceso aparte (1-2 min; mientras tanto, best sellers)…")
                    rc = self._run_script("features_cache.py", ["--build"], wait=True)
                    if rc != 0 or not fc.LIVE_CACHE_PATH.exists():
                        raise RuntimeError(f"la construcción de la caché ha fallado (código {rc}): mira logs/features_cache.out")
                    cache = FeatureCache.load(fc.LIVE_CACHE_PATH)
                    cache._upd_ts = 0.0
                    cache._ev_ts, cache.user_ev = 0.0, {}
                cache.set_oos_fast(self.unavailable)
                for name, fn in (("catálogo", cache.refresh_catalog), ("tendencias", cache.refresh_trending),
                                 ("co-views en directo", cache.refresh_live_coviews),
                                 ("perfil de usuario en directo", cache.refresh_user_events),
                                 ("compras hechas con nuestras recomendaciones", cache.load_own_purchases)):
                    try:
                        fn()
                    except Exception as e:  # noqa: BLE001
                        log(f"primer refresco de {name} fallido: {str(e)[:200]}")
                if not self.best and cache.best:
                    self.best = cache.best
                self.cache = cache
                log(f"caché en servicio: {cache.n} productos, {int(cache.avail.sum())} disponibles, "
                    f"{len(cache.new_items)} nuevos")
            except Exception as e:  # noqa: BLE001
                log(f"no se pudo preparar la caché ({str(e)[:200]}); reintento en 30 s")
                time.sleep(30)
        cache = self.cache
        self._spawn("catálogo", cache.refresh_catalog, 10)
        self._spawn("tendencias", cache.refresh_trending, 60)
        self._spawn("co-views", cache.refresh_live_coviews, 300)
        self._spawn("perfil", cache.refresh_user_events, 2)
        if AUTO_TRAIN and not self.dry_run:
            self._spawn("entrenador", self._ensure_trainer, 30)

    def _run_script(self, script: str, args: list[str], wait: bool):
        """Lanza otro script del proyecto en un proceso aparte, con prioridad baja y salida a logs/."""
        fc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = open(fc.LOG_DIR / (Path(script).stem + ".out"), "a", encoding="utf-8")
        env = dict(os.environ, TRAIN_THREADS=str(max(1, (os.cpu_count() or 2) // 2 - 1)), PYTHONIOENCODING="utf-8")
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        proc = subprocess.Popen([sys.executable, str(HERE / script), *args], cwd=str(HERE), env=env,
                                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
        if wait:
            return proc.wait()
        return proc

    def _ensure_trainer(self) -> None:
        """Mantiene vivo `train_model.py --loop`: entrena al arrancar y cada hora con el feedback nuevo.
        El propio entrenador se para solo 20 minutos antes de la final."""
        if time.time() > FINAL_START - 20 * 60 > 0 and MODEL_PATH.exists():
            return
        proc = self.trainer
        if proc is not None and proc.poll() is None:
            return
        if proc is not None and time.time() - self._trainer_started < 600:
            return                                   # murió hace poco: no relanzar en bucle
        if proc is not None:
            log(f"el entrenador terminó (código {proc.returncode}); lo relanzo. Detalle en logs/train_model.out")
        else:
            log("lanzo el entrenamiento en un proceso aparte (5-10 min la primera vez; detalle en logs/train_model.out)"
                + ("" if MODEL_PATH.exists() else " · AÚN NO HAY MODELO: hasta que acabe se responden best sellers"))
        self._trainer_started = time.time()
        self.trainer = self._run_script("train_model.py", ["--loop", str(TRAIN_EVERY_MIN)], wait=False)

    def _read_feedback(self) -> None:
        rows = fc.query(SQL_FEEDBACK_RECENT, timeout=8, name="feedback")
        cache = self.cache
        for r in rows:
            rid = str(r.get("request_id"))
            if rid in self.fb_seen:
                continue
            self.fb_seen[rid] = 1
            if len(self.fb_seen) > 50000:
                self.fb_seen.popitem(last=False)
            served = str(r.get("served_by") or "")
            clicked, purchased = fc.as_pid(r.get("clicked")), fc.as_pid(r.get("purchased"))
            meta = self.req_meta.get(rid)
            if meta is None:
                fts = fc.parse_ts(r.get("ts"))
                if fts is not None and fts > self.started + 12:
                    self.fb_window["no_vistas"] += 1          # llegó mientras corríamos y no la contestamos
                continue
            self.fb_window["n"] += 1
            self.fb_window["team"] += served == "team"
            if served == "team":
                lat = fc.as_float(r.get("latency_ms"))
                if lat == lat:
                    self.srv_latency.append(lat)
                ours = self.req_items.get(rid)
                shown = tuple(str(p) for p in fc.as_list(r.get("shown")))
                if ours is not None and shown and shown != ours:
                    self.fb_window["ajenas"] += 1             # se mostró una lista que no es la nuestra
            else:
                reason = str(r.get("fallback_reason") or "?")
                self.fb_window[f"fallback:{reason}"] += 1
                if reason == "invalid" and self.fb_window["fallback:invalid"] <= 2:
                    log(f"respuesta marcada como inválida: {rid} → {self.req_items.get(rid)}")
            user, label, cold = meta
            if served != "team":
                label = "fallback-organizador"
            for key in (label, "cold" if cold and served == "team" else None):
                if key:
                    st = self.fb_modes.setdefault(key, [0, 0, 0])
                    st[0] += 1
                    st[1] += bool(clicked)
                    st[2] += bool(purchased)
            if purchased and cache is not None:
                cache.add_purchase(user, purchased)

    def bootstrap(self) -> None:
        while not self.best:
            try:
                rows = fc.query(SQL_BEST_SELLERS, timeout=120, name="best_sellers")
                sales = sorted(((int(r["n"]), str(r["product_id"])) for r in rows), reverse=True)
                self.best = [p for _, p in sales]
                log(f"best sellers listos ({len(self.best)} productos): ya se responde en modo baseline")
            except Exception as e:  # noqa: BLE001
                log(f"no se pudieron leer los best sellers ({str(e)[:160]}); reintento en 2 s")
                if fc.LIVE_CACHE_PATH.exists():
                    try:
                        self.best = FeatureCache.load(fc.LIVE_CACHE_PATH).best
                        continue
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(2)
        try:
            self._refresh_stock()
        except Exception as e:  # noqa: BLE001
            log(f"primer chequeo de stock fallido: {str(e)[:160]}")
        try:
            self._watch_model()
        except Exception as e:  # noqa: BLE001
            log(f"modelo no cargado: {str(e)[:160]}")
        self._spawn("stock", self._refresh_stock, 1.0)
        self._spawn("modelo", self._watch_model, 20)
        self._spawn("feedback", self._read_feedback, 30)
        threading.Thread(target=self._init_cache, name="cache-init", daemon=True).start()

    # --- brazos del A/B ------------------------------------------------------------------------
    def arm_stats(self) -> dict[str, list[int]]:
        """Compras reales acumuladas de cada brazo, solo con respuestas de explotación (sin '+')."""
        agg = {"wide": [0, 0, 0], "cand": [0, 0, 0], "full": [0, 0, 0]}
        for label, st in self.fb_modes.items():
            if "+" not in label:
                arm = label.rsplit("-", 1)[-1]
                if arm in agg:
                    agg[arm] = [a + b for a, b in zip(agg[arm], st[:3])]
        return agg

    def choose_final_arm(self) -> str:
        if ARM in ("full", "cand", "wide"):
            return ARM
        if time.time() < self.slow_until:
            log("FINAL · el equipo va justo de latencia: brazo cand (retrieval rápido)")
            return "cand"
        agg = self.arm_stats()
        (nw, _, bw), (nc, _, bc) = agg["wide"], agg["cand"]
        if nw >= 3000 and nc >= 3000:
            arm = "wide" if bw / nw > bc / nc else "cand"
            log(f"FINAL · brazo elegido por conversión en vivo: {arm} "
                f"(wide {100 * bw / nw:.2f}% con n={nw}; cand {100 * bc / nc:.2f}% con n={nc})")
        else:
            arm = "wide" if (self.ranker is not None and self.ranker.wide_default) else "cand"
            log(f"FINAL · pocos datos del A/B en vivo (wide n={nw}, cand n={nc}): brazo según validación offline: {arm}")
        return arm

    def public_label(self, internal: str) -> str:
        """Nombre público corto: v7b = retrieval (~80 candidatos), v7w = retrieval amplio (~250), v7a = catálogo
        completo, v7h = control, v70 = best sellers;
        una x al final marca las respuestas de exploración. El marcador da la conversión ESPERADA de cada
        nombre (mucho menos ruidosa que nuestras compras reales): es la mejor vara de medir el A/B."""
        if internal.startswith("holdout"):
            return PUBLIC_NAME + "h"
        base = "a" if "-full" in internal else "b" if "-cand" in internal else "w" if "-wide" in internal else "0"
        return PUBLIC_NAME + base + ("x" if "+" in internal else "")

    def finalize(self, items: list[str] | None) -> list[str] | None:
        """Garantiza 10 IDs distintos, activos y en stock; completa con best sellers."""
        unavailable, cache = self.unavailable, self.cache
        out: list[str] = []
        seen: set[str] = set()

        def take(source) -> bool:
            for p in source:
                if p in seen or p in unavailable:
                    continue
                if cache is not None and cache.is_available(p) is False:
                    continue
                seen.add(p)
                out.append(p)
                if len(out) == 10:
                    return True
            return False

        if take(items or []) or take(self.best):
            return out
        if cache is not None and take(cache.pids[i] for i in np.flatnonzero(cache.avail[: cache.n]).tolist()):
            return out
        return None

    def answer(self, recs: list[dict]) -> list[dict]:
        t0 = time.time()
        final = t0 >= FINAL_START > 0
        if final and self.final_arm is None:
            self.final_arm = self.choose_final_arm()
        cache, ranker = self.cache, self.ranker
        ranked: dict[int, tuple[list[str] | None, str, object]] = {}
        todo, arms, explore = [], [], []
        for i, rec in enumerate(recs):
            rid = str(rec.get("request_id"))
            h = zlib.crc32(rid.encode())
            if not final and HOLDOUT > 0 and (h % 10000) < HOLDOUT * 10000:
                ranked[i] = (None, "holdout-baseline", None)   # grupo de control: mide el uplift real
            elif cache is None or ranker is None:
                ranked[i] = (None, "v3-bestsellers", None)
            else:
                todo.append(i)
                if final:
                    arms.append(self.final_arm)
                    explore.append(None)
                else:
                    arms.append(ARM if ARM in ("full", "cand", "wide") else ("wide" if (h >> 8) & 1 else "cand"))
                    u = self.rng.random()
                    explore.append("shuf" if u < EXPLORE_RATE / 2 else "wide" if u < EXPLORE_RATE else None)
        if len(todo) > MAX_FULL_BATCH or t0 < self.slow_until:   # atasco o equipo lento: la latencia manda
            arms = ["cand"] * len(todo)
        if todo:
            try:
                res = rank_batch(cache, ranker, [recs[i] for i in todo], arms, explore, self.rng)
            except Exception as e:  # noqa: BLE001
                log(f"fallo del ranking, respondo best sellers: {str(e)[:200]}")
                res = [(None, "v3-bestsellers", None)] * len(todo)
            for i, r in zip(todo, res):
                ranked[i] = r
        answers = []
        for i, rec in enumerate(recs):
            items, label, R = ranked[i]
            if items is None and label == "error":
                label = "v3-bestsellers"
            items = self.finalize(items)
            if items is None:
                self.stats["sin_respuesta"] += 1
                continue
            rid = str(rec.get("request_id"))
            answers.append({"request_id": rid, "items": items, "model_version": self.public_label(label)})
            self.stats[label] += 1
            cold = bool(R.cold) if R is not None else fc.as_bool(rec.get("is_new_user"), False)
            self.stats["cold"] += cold
            self.req_meta[rid] = (str(rec.get("user_id") or ""), label, cold)
            if len(self.req_meta) > 60000:
                self.req_meta.popitem(last=False)
            self.req_items[rid] = tuple(items)
            if len(self.req_items) > 20000:
                self.req_items.popitem(last=False)
        self.compute_ms.append((time.time() - t0) * 1000)
        return answers

    def push(self, answers: list[dict], recs_ts: dict[str, float]) -> None:
        """Envío en segundo plano: el bucle vuelve a leer peticiones sin esperar a que termine el insert."""
        self.pushers.submit(self._push_now, answers, recs_ts)

    def _push_now(self, answers: list[dict], recs_ts: dict[str, float]) -> None:
        try:
            self._push_sync(answers, recs_ts)
        except Exception as e:  # noqa: BLE001
            self.stats["push_perdidos"] += len(answers)
            log(f"push perdido tras reintento ({len(answers)} respuestas): {str(e)[:160]}")

    def _push_sync(self, answers: list[dict], recs_ts: dict[str, float]) -> None:
        if self.dry_run:
            log(f"[dry-run] {len(answers)} respuestas, p.ej. {answers[0]}")
        else:
            try:
                fc.insert("submissions", answers, timeout=2.5)
            except Exception as e:  # noqa: BLE001
                log(f"push fallido ({str(e)[:160]}); reintento")
                fc.insert("submissions", answers, timeout=2.5)
        now = time.time()
        self.stats["respondidas"] += len(answers)
        self.ages.extend(now - recs_ts[a["request_id"]] for a in answers if a["request_id"] in recs_ts)

    def report(self) -> None:
        if time.time() - self._last_report < 60:
            return
        self._last_report = time.time()
        ages = np.asarray(self.ages or [0.0])
        modes = {k: v for k, v in self.stats.items() if k.startswith(("v3", "holdout"))}
        line = (f"respondidas {self.stats['respondidas']} · edad al responder p50 {np.percentile(ages, 50):.2f}s "
                f"p95 {np.percentile(ages, 95):.2f}s · cómputo p50 {np.percentile(self.compute_ms or [0], 50):.0f} ms "
                f"máx {np.max(self.compute_ms or [0]):.0f} ms · cold {self.stats['cold']} · "
                f"sin stock {len(self.unavailable)} · {modes}")
        w = self.fb_window
        if w["n"]:
            fb = {k: v for k, v in w.items() if k.startswith("fallback")}
            line += f" · DE LO NUESTRO: {100 * w['team'] / w['n']:.1f}% válido y a tiempo {fb or ''}"
            if self.srv_latency:
                lat = np.asarray(self.srv_latency)
                line += (f" · latencia según el organizador p50 {np.percentile(lat, 50):.0f} ms "
                         f"p95 {np.percentile(lat, 95):.0f} ms")
        log(line)
        # guardia de latencia: si el cómputo o la latencia medida suben, el catálogo completo se apaga 15 min
        comp50 = float(np.percentile(self.compute_ms or [0], 50))
        lat95 = float(np.percentile(self.srv_latency, 95)) if len(self.srv_latency) >= 20 else 0.0
        if ARM != "cand" and (comp50 > SLOW_COMPUTE_MS or lat95 > SLOW_P95_MS):
            if time.time() >= self.slow_until:
                log(f"   LATENCIA ALTA (cómputo p50 {comp50:.0f} ms, p95 organizador {lat95:.0f} ms): "
                    f"paso todo a retrieval rápido durante 15 min")
            self.slow_until = time.time() + 900
        elif lat95 > SLOW_P95_MS:
            log(f"   LATENCIA ALTA (p95 organizador {lat95:.0f} ms) ya en modo rápido: es la red o el equipo, no el modelo")
        if w["no_vistas"]:
            log(f"   ATENCIÓN: {w['no_vistas']} peticiones llegaron y no las contestamos (bucle parado o red caída)")
        if w["ajenas"]:
            log(f"   ATENCIÓN: en {w['ajenas']} peticiones se mostró una lista que NO es la nuestra: hay OTRO proceso "
                f"respondiendo con la clave del equipo. Paradlo: se pisan y falsea el marcador.")
        if self.ranker is None:
            log("   SIN MODELO: se están respondiendo best sellers (= baseline). "
                + ("El entrenamiento está en marcha; se cargará solo." if self.trainer is not None
                   else "Hace falta ejecutar train_model.py."))
        self.srv_latency = []
        warn = self.consistency_warning()
        if warn:
            log(warn)
        if self.fb_modes:
            agg = self.arm_stats()
            groups = {"A/B wide": agg["wide"], "A/B cand": agg["cand"], "full": agg["full"]}
            for key in ("holdout-baseline", "cold", "fallback-organizador"):
                if key in self.fb_modes:
                    groups[key] = self.fb_modes[key]
            for suffix in ("+shuf", "+wide"):
                st = [0, 0, 0]
                for label, v in self.fb_modes.items():
                    if label.endswith(suffix):
                        st = [a + b for a, b in zip(st, v)]
                groups["explora " + suffix] = st
            log("   conversión real acumulada: " + " | ".join(
                f"{k}: n={n} ctr={100 * c / n:.1f}% conv={100 * b / n:.2f}%" for k, (n, c, b) in groups.items() if n))
            try:
                STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                tmp = STATE_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.fb_modes))
                os.replace(tmp, STATE_PATH)
            except Exception:  # noqa: BLE001
                pass
        self.stats = Counter({k: v for k, v in self.stats.items() if k == "respondidas"})
        self.ages, self.compute_ms, self.fb_window = [], [], Counter()

    def pull(self, seen: OrderedDict) -> tuple[list[dict], dict[str, float]]:
        sqls = (SQL_NEW_REQUESTS_FAST, SQL_NEW_REQUESTS, SQL_NEW_REQUESTS_ALT)
        try:
            rows = fc.query(sqls[self._req_sql], timeout=3, name="requests")
            self._pull_fail = 0
        except Exception:
            self._pull_fail += 1
            if self._pull_fail == 3:   # la query no funciona: se pasa a la siguiente variante
                self._req_sql = (self._req_sql + 1) % len(sqls)
                self._pull_fail = 0
                log(f"lectura de peticiones: cambio a la variante {self._req_sql}")
            raise
        if self._req_sql == 0:         # la ventana corta devuelve vacío: ¿no hay tráfico o falla el filtro?
            self._empty = 0 if rows else self._empty + 1
            if self._empty >= 12:
                self._empty = 0
                rows = fc.query(SQL_NEW_REQUESTS, timeout=3, name="requests")
                newest = fc.parse_ts(rows[0].get("ts")) if rows else None
                if newest is not None and time.time() - newest < 5:
                    self._req_sql = 1
                    log("lectura de peticiones: la ventana corta no devuelve datos; uso la lectura completa")
        now = time.time()
        fresh, ts_map = [], {}
        for r in reversed(rows):          # de la más antigua a la más nueva
            rid = str(r.get("request_id"))
            if rid in seen:
                continue
            seen[rid] = 1
            if len(seen) > 30000:
                seen.popitem(last=False)
            ts = fc.parse_ts(r.get("ts"))
            if ts is not None:
                if now - ts > MAX_AGE:
                    self.stats["descartadas-viejas"] += 1
                    continue
                ts_map[rid] = ts
            fresh.append(r)
        return fresh, ts_map

    def run(self) -> None:
        self.bootstrap()
        seen: OrderedDict[str, int] = OrderedDict()
        log(f"bucle en marcha · equipo {fc.TEAM} · brazo {ARM} · exploración {EXPLORE_RATE} · holdout {HOLDOUT} · "
            f"final desde {fc.fmt_ts(FINAL_START)} UTC" + (" · DRY-RUN" if self.dry_run else ""))
        while True:
            t0 = time.time()
            if t0 - self._last_loop > 5:
                log(f"¡¡ EL BUCLE HA ESTADO PARADO {t0 - self._last_loop:.0f} s !! (equipo suspendido, red caída o "
                    f"proceso congelado): esas peticiones se han perdido")
            self._last_loop = t0
            try:
                fresh, ts_map = self.pull(seen)
                if fresh:
                    answers = self.answer(fresh)
                    if answers:
                        self.push(answers, ts_map)
                self.report()
            except Exception as e:  # noqa: BLE001  seguir vivos: los compradores llegan mientras dormimos
                log(f"error en el bucle: {str(e)[:200]}")
                time.sleep(0.3)
            time.sleep(max(0.03, POLL - (time.time() - t0)))


def keep_awake() -> None:
    """Windows: impide que el equipo entre en suspensión mientras el recomendador corre."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)
        log("suspensión automática de Windows desactivada mientras este proceso esté en marcha")
    except Exception:  # noqa: BLE001  (no es Windows)
        pass


if __name__ == "__main__":
    engine = Engine(dry_run="--dry-run" in sys.argv)
    keep_awake()
    try:
        engine.run()
    except KeyboardInterrupt:
        log("parado por el usuario")
    finally:
        if engine.trainer is not None and engine.trainer.poll() is None:
            engine.trainer.terminate()
