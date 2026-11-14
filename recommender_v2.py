"""recommender_v2.py · recomendador de producción en dos fases (team_40).

    python recommender_v2.py             # producción
    python recommender_v2.py --dry-run   # calcula respuestas pero NO las inserta (para probar)

Bucle principal (pull → predict → push), igual que el baseline pero:
  1. Retrieval en RAM (features_cache.FeatureCache): vistas recientes, co-views, view→purchase,
     vecinos de texto, tendencias, top por categoría… → ~50-90 candidatos en stock.
  2. Re-ranking con LightGBM (models/ranker.pkl, entrenado por train_model.py):
     P(compra) = P(clic) × P(compra | clic). Todo el lote se puntúa en una sola llamada.
  3. Validación final: exactamente 10 IDs distintos, activos y con stock.

Degradación en cascada (nunca se deja una petición sin respuesta válida):
     modelo LightGBM → ranking heurístico → best sellers en stock (= baseline).
El proceso arranca respondiendo en modo baseline en cuanto tiene los best sellers, y sube de nivel
solo cuando la caché y el modelo están cargados (en segundo plano). Todo lo lento (stock cada 1 s,
catálogo, tendencias, co-views en directo, feedback, recarga en caliente del modelo) va en hilos:
el bucle principal solo hace 2 llamadas HTTP por iteración.

Variables de entorno opcionales: EXPLORE_EPS (0.03), HOLDOUT (0.02), REBUY_PENALTY (0.3),
FINAL_START_UTC (2026-11-15T09:00:00Z: desde entonces exploración y holdout se apagan solos),
MODEL_PATH, CACHE_DIR, POLL_SECONDS (0.25).
"""
from __future__ import annotations

import os
import pickle
import random
import sys
import threading
import time
import zlib
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

import features_cache as fc
from features_cache import F, FeatureCache, log

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE / "models" / "ranker.pkl"))
POLL = float(os.environ.get("POLL_SECONDS", 0.25))
MAX_AGE = 6.0            # peticiones más viejas que esto ya han recibido el fallback: no se contestan
EXPLORE_EPS = float(os.environ.get("EXPLORE_EPS", 0.03))
HOLDOUT = float(os.environ.get("HOLDOUT", 0.02))
REBUY_PENALTY = float(os.environ.get("REBUY_PENALTY", 0.3))
FINAL_START = fc.parse_ts(os.environ.get("FINAL_START_UTC", "2026-11-15T09:00:00Z")) or 0.0

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

SQL_FEEDBACK_RECENT = f"""
SELECT request_id, served_by, fallback_reason, clicked, purchased, latency_ms
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
        self.click = lgb.Booster(model_str=bundle["click"])
        use_buy = bundle.get("buy") and bundle.get("combine") == "click_x_buy"
        self.buy = lgb.Booster(model_str=bundle["buy"]) if use_buy else None
        self.vocab = bundle.get("vocab", {})
        self.version = str(bundle.get("version", "lgb"))

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "Ranker":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def score(self, X: np.ndarray) -> np.ndarray:
        p = self.click.predict(X, num_threads=1)
        if self.buy is not None:
            p = p * self.buy.predict(X, num_threads=1)
        return np.nan_to_num(np.asarray(p, np.float64), nan=0.0)


def heuristic_score(X: np.ndarray) -> np.ndarray:
    """Ranking sin modelo: combinación lineal de las señales más fiables. Es el plan B."""
    return (2.0 * X[:, F["view_recency"]]
            + 6.0 * X[:, F["buy_score"]]
            + 4.0 * X[:, F["cov_score"]]
            + 1.5 * np.maximum(X[:, F["text_max"]], 0)
            + 0.5 * X[:, F["same_sub_last"]]
            + 0.2 * X[:, F["same_cat_last"]]
            + 0.5 * X[:, F["user_cat_aff"]]
            + 0.25 * X[:, F["pop_purch"]]
            + 20.0 * X[:, F["car_cvr"]]
            + 0.15 * np.minimum(X[:, F["trend_lift"]], 3.0)).astype(np.float64)


def rank_batch(cache: FeatureCache, ranker: Ranker | None, recs: list[dict], explore_eps: float = 0.0,
               rng: random.Random | None = None) -> list[tuple[list[str] | None, str, object]]:
    """Dos fases para un lote de peticiones → [(10 product_ids | None, modo, Request | None)]."""
    prepared: list[tuple | None] = []
    vocab = ranker.vocab if ranker is not None else None
    for rec in recs:
        try:
            R = cache.parse_request(rec)
            S = cache.session(R)
            cands = cache.candidates(R, S)
            prepared.append((R, cands, cache.featurize(R, S, cands, vocab)))
        except Exception as e:  # noqa: BLE001  una petición rara no tumba el lote
            log(f"petición {rec.get('request_id')} no procesable: {str(e)[:150]}")
            prepared.append(None)
    ok = [p for p in prepared if p is not None]
    scores, mode = None, "v2-heur"
    if ok:
        X_all = np.vstack([p[2] for p in ok])
        if ranker is not None:
            try:
                scores = ranker.score(X_all)
                mode = f"v2-{ranker.version}"
            except Exception as e:  # noqa: BLE001
                log(f"el modelo ha fallado, uso el heurístico: {str(e)[:150]}")
        if scores is None:
            scores = heuristic_score(X_all)
    out: list[tuple[list[str] | None, str, object]] = []
    off = 0
    pids = cache.pids
    for p in prepared:
        if p is None:
            out.append((None, "error", None))
            continue
        R, cands, X = p
        s = np.array(scores[off: off + len(cands)], np.float64)
        off += len(cands)
        s[X[:, F["user_bought"]] > 0] *= REBUY_PENALTY   # ya lo compró: difícilmente repite
        order = np.argsort(-s, kind="stable")
        top = cands[order[:10]].tolist()
        m = mode
        if explore_eps > 0 and rng is not None and len(order) > 12 and rng.random() < explore_eps:
            pool = cands[order[10:25]].tolist()            # exploración: el último slot prueba otro candidato
            fresh = [i for i in pool if i in cache.new_items]
            top[-1] = rng.choice(fresh or pool)
            m = mode + "+x"
        out.append(([pids[i] for i in top], m, R))
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
        self.req_meta: OrderedDict[str, tuple[str, str, bool]] = OrderedDict()  # request_id → (user, modo, cold)
        self.fb_seen: OrderedDict[str, int] = OrderedDict()
        self.fb_modes: dict[str, list[int]] = {}   # modo → [peticiones, clics, compras]
        self.fb_window: Counter = Counter()
        self._last_report = time.time()
        self._req_sql = SQL_NEW_REQUESTS
        self._pull_fail = 0

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
        self._model_mtime = mtime          # si el fichero está corrupto no se reintenta hasta que cambie
        ranker = Ranker.load(MODEL_PATH)
        ranker.score(np.zeros((2, fc.NF), np.float32))  # prueba de humo antes de publicarlo
        self.ranker = ranker
        log(f"modelo cargado: {ranker.version} ({'clic×compra' if ranker.buy is not None else 'clic'})")

    def _init_cache(self) -> None:
        while self.cache is None:
            try:
                cache = None
                if fc.LIVE_CACHE_PATH.exists():
                    try:
                        cache = FeatureCache.load(fc.LIVE_CACHE_PATH)
                        cache._upd_ts = 0.0               # fuerza una lectura completa de catalog_updates
                        cache._purchases_loaded = False
                        log(f"caché cargada de disco ({cache.n} productos)")
                    except Exception as e:  # noqa: BLE001
                        log(f"caché de disco no válida ({str(e)[:120]}): la reconstruyo")
                if cache is None:
                    log("construyendo la caché de features en segundo plano (mientras tanto: modo baseline)…")
                    cache = FeatureCache.build()
                    cache.save(fc.LIVE_CACHE_PATH)
                cache.set_oos_fast(self.unavailable)
                for name, fn in (("catálogo", cache.refresh_catalog), ("tendencias", cache.refresh_trending),
                                 ("co-views en directo", cache.refresh_live_coviews),
                                 ("compras en directo", cache.refresh_live_purchases)):
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
        self._spawn("compras", cache.refresh_live_purchases, 60)

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
            self.fb_window["n"] += 1
            self.fb_window["team"] += served == "team"
            if served != "team":
                self.fb_window[f"fallback:{r.get('fallback_reason') or '?'}"] += 1
            meta = self.req_meta.get(rid)
            if meta is None:
                continue
            user, mode, cold = meta
            if served != "team":
                mode = "fallback-organizador"
            for key in (mode, "cold" if cold and served == "team" else None):
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
        self._spawn("stock", self._refresh_stock, 1.0)
        self._spawn("modelo", self._watch_model, 20)
        self._spawn("feedback", self._read_feedback, 30)
        threading.Thread(target=self._init_cache, name="cache-init", daemon=True).start()

    # --- respuesta ----------------------------------------------------------------------------
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
        eps = 0.0 if final else EXPLORE_EPS
        holdout = 0.0 if final else HOLDOUT
        cache = self.cache
        ranked: dict[int, tuple[list[str] | None, str, object]] = {}
        todo: list[int] = []
        for i, rec in enumerate(recs):
            rid = str(rec.get("request_id"))
            if holdout > 0 and (zlib.crc32(rid.encode()) % 10000) < holdout * 10000:
                ranked[i] = (None, "holdout-baseline", None)   # grupo de control: mide el uplift real
            elif cache is None:
                ranked[i] = (None, "v2-bestsellers", None)
            else:
                todo.append(i)
        if todo:
            try:
                res = rank_batch(cache, self.ranker, [recs[i] for i in todo], eps, self.rng)
            except Exception as e:  # noqa: BLE001
                log(f"fallo del ranking, respondo best sellers: {str(e)[:200]}")
                res = [(None, "v2-bestsellers", None)] * len(todo)
            for i, r in zip(todo, res):
                ranked[i] = r
        answers = []
        for i, rec in enumerate(recs):
            items, mode, R = ranked[i]
            if items is None and mode == "error":
                mode = "v2-bestsellers"
            items = self.finalize(items)
            if items is None:
                self.stats["sin_respuesta"] += 1
                continue
            rid = str(rec.get("request_id"))
            answers.append({"request_id": rid, "items": items, "model_version": mode})
            self.stats[mode] += 1
            cold = bool(R.cold) if R is not None else fc.as_bool(rec.get("is_new_user"), False)
            self.stats["cold"] += cold
            self.req_meta[rid] = (str(rec.get("user_id") or ""), mode, cold)
            if len(self.req_meta) > 60000:
                self.req_meta.popitem(last=False)
        self.compute_ms.append((time.time() - t0) * 1000)
        return answers

    def push(self, answers: list[dict], recs_ts: dict[str, float]) -> None:
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
        modes = {k: v for k, v in self.stats.items() if k.startswith(("v2", "holdout"))}
        line = (f"respondidas {self.stats['respondidas']} · edad al responder p50 {np.percentile(ages, 50):.2f}s "
                f"p95 {np.percentile(ages, 95):.2f}s · cómputo medio {np.mean(self.compute_ms or [0]):.1f} ms · "
                f"modos {modes} · cold {self.stats['cold']} · sin stock {len(self.unavailable)}")
        w = self.fb_window
        if w["n"]:
            fb = {k: v for k, v in w.items() if k.startswith("fallback")}
            line += f" · feedback: {100 * w['team'] / w['n']:.1f}% a tiempo {fb or ''}"
        log(line)
        if self.fb_modes:
            log("   conversión en vivo por modo: " + " | ".join(
                f"{k}: n={n} ctr={100 * c / max(n, 1):.1f}% conv={100 * b / max(n, 1):.2f}%"
                for k, (n, c, b) in sorted(self.fb_modes.items())))
        self.stats = Counter({k: v for k, v in self.stats.items() if k == "respondidas"})
        self.ages, self.compute_ms, self.fb_window = [], [], Counter()

    def pull(self, seen: OrderedDict) -> tuple[list[dict], dict[str, float]]:
        try:
            rows = fc.query(self._req_sql, timeout=3, name="requests")
            self._pull_fail = 0
        except Exception:
            self._pull_fail += 1
            if self._pull_fail == 3:   # por si alguna columna no existe: lectura genérica
                self._req_sql = SQL_NEW_REQUESTS_ALT if self._req_sql == SQL_NEW_REQUESTS else SQL_NEW_REQUESTS
                self._pull_fail = 0
            raise
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
                    self.stats["v2-descartadas-viejas"] += 1
                    continue
                ts_map[rid] = ts
            fresh.append(r)
        return fresh, ts_map

    def run(self) -> None:
        self.bootstrap()
        seen: OrderedDict[str, int] = OrderedDict()
        log(f"bucle en marcha · equipo {fc.TEAM} · exploración {EXPLORE_EPS} · holdout {HOLDOUT}"
            + (" · DRY-RUN" if self.dry_run else ""))
        while True:
            t0 = time.time()
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
            time.sleep(max(0.05, POLL - (time.time() - t0)))


if __name__ == "__main__":
    try:
        Engine(dry_run="--dry-run" in sys.argv).run()
    except KeyboardInterrupt:
        log("parado por el usuario")
