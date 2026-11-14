"""recommender_v2.py · Recomendador v4-apex (team_40): ~100% On-Time + Máxima Conversión y Cold Start.

    python recommender_v2.py             # Producción
    python recommender_v2.py --dry-run   # Prueba sin insertar
"""
from __future__ import annotations

import os
import pickle
import random
import sys
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

import features_cache as fc
from features_cache import F, FeatureCache, log

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE / "models" / "ranker.pkl"))
POLL = float(os.environ.get("POLL_SECONDS", 0.12))
MAX_AGE = 2.65
EXPLORE_EPS = float(os.environ.get("EXPLORE_EPS", 0.015))
REBUY_PENALTY = float(os.environ.get("REBUY_PENALTY", 0.25))
REVENUE_ALPHA = float(os.environ.get("REVENUE_ALPHA", 0.04))   # Tie-breaker suave: no distorsiona el Top 1
FINAL_START = fc.parse_ts(os.environ.get("FINAL_START_UTC", "2026-11-15T09:00:00Z")) or 0.0

SQL_BEST_SELLERS = """
SELECT product_id::String AS product_id, count() AS n
FROM shop.history_events
WHERE event_type = 'purchase'
GROUP BY product_id
ORDER BY n DESC, product_id DESC
LIMIT 200"""

SQL_NEW_REQUESTS_FAST = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
WHERE ts > now() - INTERVAL 8 SECOND
ORDER BY ts DESC
LIMIT 45"""

SQL_NEW_REQUESTS_FALLBACK = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
ORDER BY ts DESC
LIMIT 45"""

SQL_FEEDBACK_RECENT = f"""
SELECT request_id, shown, served_by, fallback_reason, clicked, purchased, latency_ms, revenue_eur
FROM {fc.TEAM}.feedback
WHERE ts > now() - INTERVAL 3 MINUTE AND request_id != '__init__'"""


# ----------------------------------------------------------------------------------------------
# Ranking v4-apex (Conversión Pura + Live Stream Boost + Cold Start Inteligente)
# ----------------------------------------------------------------------------------------------
class Ranker:
    def __init__(self, bundle: dict) -> None:
        import lightgbm as lgb

        if bundle.get("features") != fc.FEATURES:
            raise RuntimeError("el modelo se entrenó con otra lista de features: vuelve a entrenar")
        self.click = lgb.Booster(model_str=bundle["click"])
        use_buy = bundle.get("buy") and bundle.get("combine") in ("click_x_buy", "ensemble")
        self.buy = lgb.Booster(model_str=bundle["buy"]) if use_buy else None
        self.direct_buy = lgb.Booster(model_str=bundle["direct_buy"]) if bundle.get("direct_buy") else None
        self.vocab = bundle.get("vocab", {})
        self.version = str(bundle.get("version", "lgb"))

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "Ranker":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def score(self, X: np.ndarray) -> np.ndarray:
        p_click = np.clip(self.click.predict(X, num_threads=2), 1e-6, 1.0)
        if self.buy is not None:
            p_cond = np.clip(self.buy.predict(X, num_threads=2), 1e-6, 1.0)
            # P(compra) = P(clic) * P(compra | clic)^0.85 (suaviza la varianza de pocas compras)
            p = p_click * (p_cond ** 0.85)
        else:
            p = p_click
        if self.direct_buy is not None:
            p_dir = np.clip(self.direct_buy.predict(X, num_threads=2), 1e-6, 1.0)
            # Media geométrica ponderada: preserva probabilidades calibradas sin que una domine
            p = (p ** 0.65) * (p_dir ** 0.35)
        return np.nan_to_num(np.asarray(p, np.float64), nan=0.0)


def heuristic_score(X: np.ndarray) -> np.ndarray:
    return (2.2 * X[:, F["view_recency"]]
            + 7.0 * X[:, F["buy_score"]]
            + 4.5 * X[:, F["cov_score"]]
            + 1.6 * np.maximum(X[:, F["text_max"]], 0)
            + 0.6 * X[:, F["same_sub_last"]]
            + 0.25 * X[:, F["same_cat_last"]]
            + 0.6 * X[:, F["user_cat_aff"]]
            + 0.28 * X[:, F["pop_purch"]]
            + 22.0 * X[:, F["car_cvr"]]
            + 0.20 * np.minimum(X[:, F["trend_lift"]], 4.0)).astype(np.float64)


def rank_batch(cache: FeatureCache, ranker: Ranker | None, recs: list[dict], explore_eps: float = 0.0,
               rng: random.Random | None = None) -> list[tuple[list[str] | None, str, object]]:
    prepared: list[tuple | None] = []
    vocab = ranker.vocab if ranker is not None else None
    for rec in recs:
        try:
            R = cache.parse_request(rec)
            S = cache.session(R)
            cands = cache.candidates(R, S)
            prepared.append((R, S, cands, cache.featurize(R, S, cands, vocab)))
        except Exception as e:  # noqa: BLE001
            log(f"petición {rec.get('request_id')} no procesable: {str(e)[:120]}")
            prepared.append(None)

    ok = [p for p in prepared if p is not None]
    scores_all, mode = None, "v4-heur"
    if ok:
        X_all = np.vstack([p[3] for p in ok])
        if ranker is not None:
            try:
                scores_all = ranker.score(X_all)
                mode = f"v4-{ranker.version}"
            except Exception as e:  # noqa: BLE001
                log(f"fallo modelo LightGBM, uso heurístico: {str(e)[:120]}")
        if scores_all is None:
            scores_all = heuristic_score(X_all)

    out: list[tuple[list[str] | None, str, object]] = []
    off = 0
    pids = cache.pids
    price_drop = cache.price_drop
    live_buy = cache.live_buy_boost
    prices = cache.price

    for p in prepared:
        if p is None:
            out.append((None, "error", None))
            continue
        R, S, cands, X = p
        k = len(cands)
        s = np.array(scores_all[off: off + k], np.float64)
        off += k

        # 1. Multiplicador en tiempo real por Rebaja de Precio (Price Drop anunciado en catalog_updates)
        if price_drop is not None:
            drops = price_drop[cands]
            # Si un producto bajó de precio, los compradores del simulador compran mucho más
            s *= (1.0 + 1.10 * np.maximum(drops, 0.0))
            # Si subió mucho de precio, ligera penalización
            s *= (1.0 + 0.30 * np.minimum(drops, 0.0))

        # 2. Multiplicador por Compras en Directo (Tendencia en shop.events últimos 90 min)
        if live_buy is not None:
            s *= (1.0 + 0.18 * live_buy[cands])

        # 3. Cold Start Booster (Cuando el usuario ve un producto nuevo o no tiene historial):
        #    Los árboles de LightGBM a veces infra-puntúan is_new_item=1; aquí usamos similitud de texto y co-views vivos
        if R.cold:
            new_mask = X[:, F["is_new_item"]] > 0.5
            if new_mask.any():
                sim_boost = 1.0 + 0.35 * np.maximum(X[:, F["text_max"]], 0.0) + 0.25 * (X[:, F["cov_score"]] > 0)
                s = np.where(new_mask, s * sim_boost, s)

        # 4. Tie-breaker suave por Expected Revenue (solo desempata probabilidades cercanas)
        if REVENUE_ALPHA > 0:
            cand_prices = np.maximum(prices[cands], 1.0)
            rev_mult = 1.0 + REVENUE_ALPHA * np.clip(np.log1p(cand_prices) / 4.0, 0.0, 1.0)
            if S.mean_price > 0:
                rev_mult = np.where(cand_prices > 3.0 * S.mean_price, 1.0, rev_mult)
            s *= rev_mult

        # 5. Penalizar productos que este usuario ya compró
        s[X[:, F["user_bought"]] > 0] *= REBUY_PENALTY

        order = np.argsort(-s, kind="stable")
        top = cands[order[:10]].tolist()

        m = mode
        if explore_eps > 0 and rng is not None and len(order) > 12 and rng.random() < explore_eps:
            pool = cands[order[10:22]].tolist()
            fresh = [i for i in pool if i in cache.new_items]
            top[-1] = rng.choice(fresh or pool)
            m = mode + "+x"
        out.append(([pids[i] for i in top], m, R))
    return out


# ----------------------------------------------------------------------------------------------
# Motor de Producción
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
        self.req_meta: OrderedDict[str, tuple[str, str, bool]] = OrderedDict()
        self.fb_seen: OrderedDict[str, int] = OrderedDict()
        self.fb_modes: dict[str, list[float]] = {}
        self.fb_window: Counter = Counter()
        self._last_report = time.time()
        self._use_fast_sql = True
        self._clock_offset = 0.0

    def _refresh_stock(self) -> None:
        rows = fc.query(fc.SQL_OUT_OF_STOCK, timeout=2.5, name="out_of_stock")
        pids = {str(r["product_id"]) for r in rows}
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
        self._model_mtime = mtime
        ranker = Ranker.load(MODEL_PATH)
        ranker.score(np.zeros((2, fc.NF), np.float32))
        self.ranker = ranker
        log(f"modelo cargado en caliente: {ranker.version}")

    def _read_feedback(self) -> None:
        rows = fc.query(SQL_FEEDBACK_RECENT, timeout=4.0, name="feedback")
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
            rev = fc.as_float(r.get("revenue_eur"), 0.0)
            self.fb_window["n"] += 1
            self.fb_window["team"] += served == "team"
            if served != "team":
                self.fb_window[f"fallback:{r.get('fallback_reason') or '?'}"] += 1

            # Aprendizaje continuo del feedback: actualiza CTR/CVR de productos mostrados
            if cache is not None:
                get = cache.pid2idx.get
                shown_idxs = [get(str(p)) for p in fc.as_list(r.get("shown"))]
                shown_idxs = [i for i in shown_idxs if i is not None]
                c_idx = get(clicked) if clicked else None
                b_idx = get(purchased) if purchased else None
                if shown_idxs:
                    cache.update_feedback_stats(shown_idxs, c_idx, b_idx)

            meta = self.req_meta.get(rid)
            if meta is None:
                continue
            user, mode, cold = meta
            if served != "team":
                mode = "fallback-organizador"
            for key in (mode, "cold" if cold and served == "team" else None):
                if key:
                    st = self.fb_modes.setdefault(key, [0, 0, 0, 0.0])
                    st[0] += 1
                    st[1] += bool(clicked)
                    st[2] += bool(purchased)
                    st[3] += rev
            if purchased and cache is not None:
                cache.add_purchase(user, purchased)

    def _background_scheduler(self) -> None:
        last_cat = 0.0
        last_model = 0.0
        last_fb = 0.0
        last_trend = 0.0
        last_purch = 0.0
        last_cov = 0.0

        while True:
            try:
                self._refresh_stock()
            except Exception as e:  # noqa: BLE001
                log(f"[bg:stock] {str(e)[:140]}")

            time.sleep(0.2)
            now = time.time()

            if now - last_model >= 10.0:
                last_model = now
                try:
                    self._watch_model()
                except Exception as e:  # noqa: BLE001
                    log(f"[bg:model] {str(e)[:140]}")

            if self.cache is not None and now - last_cat >= 8.0:
                last_cat = now
                try:
                    self.cache.refresh_catalog()
                except Exception as e:  # noqa: BLE001
                    log(f"[bg:catalog] {str(e)[:140]}")
                time.sleep(0.15)

            if now - last_fb >= 15.0:
                last_fb = now
                try:
                    self._read_feedback()
                except Exception as e:  # noqa: BLE001
                    log(f"[bg:feedback] {str(e)[:140]}")
                time.sleep(0.15)

            if self.cache is not None and now - last_trend >= 35.0:
                last_trend = now
                try:
                    self.cache.refresh_trending()
                except Exception as e:  # noqa: BLE001
                    log(f"[bg:trending] {str(e)[:140]}")
                time.sleep(0.15)

            if self.cache is not None and now - last_purch >= 60.0:
                last_purch = now
                try:
                    self.cache.refresh_live_purchases()
                except Exception as e:  # noqa: BLE001
                    log(f"[bg:purchases] {str(e)[:140]}")
                time.sleep(0.15)

            if self.cache is not None and now - last_cov >= 120.0:
                last_cov = now
                try:
                    n_cov = self.cache.refresh_live_coviews()
                    if n_cov > 0 and int(now) % 360 < 130:
                        log(f"[bg:coviews] actualizados {n_cov} productos con co-views/compras en vivo")
                except Exception as e:  # noqa: BLE001
                    log(f"[bg:coviews] {str(e)[:140]}")

            time.sleep(0.8)

    def bootstrap(self) -> None:
        if fc.LIVE_CACHE_PATH.exists():
            try:
                log(f"cargando caché desde {fc.LIVE_CACHE_PATH}…")
                cache = FeatureCache.load(fc.LIVE_CACHE_PATH)
                self.cache = cache
                self.best = list(cache.best)
                log(f"caché cargada ({cache.n} productos)")
            except Exception as e:  # noqa: BLE001
                log(f"error cargando caché ({e})")

        try:
            self._watch_model()
        except Exception as e:  # noqa: BLE001
            log(f"aviso al cargar modelo inicial: {e}")

        while not self.best:
            try:
                rows = fc.query(SQL_BEST_SELLERS, timeout=30, name="best_sellers")
                self.best = [str(r["product_id"]) for r in rows]
                log(f"best sellers listos ({len(self.best)} productos)")
            except Exception as e:  # noqa: BLE001
                log(f"reintentando best_sellers en 1s ({str(e)[:120]})")
                time.sleep(1.0)

        try:
            self._refresh_stock()
        except Exception as e:  # noqa: BLE001
            log(f"aviso stock inicial: {e}")

        if self.cache is not None:
            for name, fn in (("catálogo", self.cache.refresh_catalog),
                             ("tendencias", self.cache.refresh_trending),
                             ("co-views en vivo", self.cache.refresh_live_coviews)):
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    log(f"aviso {name} inicial: {e}")
            log(f"caché lista en vivo: {self.cache.n} prods ({len(self.cache.new_items)} nuevos, "
                f"{len(self.cache.cov_live)} con co-views en vivo)")

        threading.Thread(target=self._background_scheduler, name="bg-scheduler", daemon=True).start()

    def finalize(self, items: list[str] | None) -> list[str] | None:
        unavailable, cache = self.unavailable, self.cache
        out: list[str] = []
        seen: set[str] = set()

        def take(source) -> bool:
            for p in source:
                if not p or p in seen or p in unavailable:
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
        cache = self.cache

        if cache is None:
            res = [(None, "v4-bestsellers", None)] * len(recs)
        else:
            try:
                res = rank_batch(cache, self.ranker, recs, eps, self.rng)
            except Exception as e:  # noqa: BLE001
                log(f"fallo ranking ({str(e)[:140]}), uso fallback interno")
                res = [(None, "v4-bestsellers", None)] * len(recs)

        answers = []
        for rec, (items, mode, R) in zip(recs, res):
            if items is None and mode == "error":
                mode = "v4-bestsellers"
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
        self.compute_ms.append((time.time() - t0) * 1000.0)
        return answers

    def push(self, answers: list[dict], recs_ts: dict[str, float]) -> None:
        if self.dry_run:
            log(f"[dry-run] {len(answers)} respuestas · ej: {answers[0]}")
        else:
            try:
                fc.insert("submissions", answers, timeout=2.2)
            except Exception as e:  # noqa: BLE001
                log(f"push reintento ({str(e)[:120]})")
                fc.insert("submissions", answers, timeout=2.2)
        now = time.time() - self._clock_offset
        self.stats["respondidas"] += len(answers)
        self.ages.extend(max(0.0, now - recs_ts[a["request_id"]]) for a in answers if a["request_id"] in recs_ts)

    def pull(self, seen: OrderedDict) -> tuple[list[dict], dict[str, float]]:
        sql = SQL_NEW_REQUESTS_FAST if self._use_fast_sql else SQL_NEW_REQUESTS_FALLBACK
        try:
            rows = fc.query(sql, timeout=2.2, name="requests")
        except Exception:
            self._use_fast_sql = not self._use_fast_sql
            raise

        now_local = time.time()
        if rows:
            newest_ts = fc.parse_ts(rows[0].get("ts"))
            if newest_ts is not None:
                obs_offset = now_local - newest_ts
                if abs(obs_offset) > 0.5:
                    self._clock_offset = 0.9 * self._clock_offset + 0.1 * (obs_offset - 0.15)

        now_srv = now_local - self._clock_offset
        fresh, ts_map = [], {}
        for r in rows:
            rid = str(r.get("request_id"))
            if rid in seen:
                continue
            seen[rid] = 1
            if len(seen) > 30000:
                seen.popitem(last=False)
            ts = fc.parse_ts(r.get("ts"))
            if ts is not None:
                if now_srv - ts > MAX_AGE:
                    self.stats["descartadas_viejas"] += 1
                    continue
                ts_map[rid] = ts
            fresh.append(r)
        return fresh, ts_map

    def report(self) -> None:
        if time.time() - self._last_report < 30:
            return
        self._last_report = time.time()
        ages = np.asarray(self.ages or [0.0])
        modes = {k: v for k, v in self.stats.items() if k.startswith("v4")}
        line = (f"resp {self.stats['respondidas']} · latencia p50={np.percentile(ages, 50):.2f}s "
                f"p95={np.percentile(ages, 95):.2f}s · rank_ms={np.mean(self.compute_ms or [0]):.1f}ms · "
                f"modos={modes} · oos={len(self.unavailable)}")
        w = self.fb_window
        if w["n"]:
            fb = {k: v for k, v in w.items() if k.startswith("fallback")}
            line += f" · ON-TIME={100 * w['team'] / w['n']:.1f}% {fb or ''}"
        log(line)
        if self.fb_modes:
            log("   vivo: " + " | ".join(
                f"{k}: n={int(n)} ctr={100 * c / max(n, 1):.2f}% conv={100 * b / max(n, 1):.2f}% rev=€{rv / max(n, 1):.3f}"
                for k, (n, c, b, rv) in sorted(self.fb_modes.items())))
        self.stats = Counter({k: v for k, v in self.stats.items() if k == "respondidas"})
        self.ages, self.compute_ms, self.fb_window = [], [], Counter()

    def run(self) -> None:
        self.bootstrap()
        seen: OrderedDict[str, int] = OrderedDict()
        log(f"Motor v4-apex activo · equipo {fc.TEAM} · poll={POLL}s · rev_alpha={REVENUE_ALPHA}")
        while True:
            t0 = time.time()
            try:
                fresh, ts_map = self.pull(seen)
                if fresh:
                    answers = self.answer(fresh)
                    if answers:
                        self.push(answers, ts_map)
                self.report()
            except Exception as e:  # noqa: BLE001
                log(f"error en bucle principal: {str(e)[:160]}")
                time.sleep(0.1)
            elapsed = time.time() - t0
            if elapsed < POLL:
                time.sleep(POLL - elapsed)


if __name__ == "__main__":
    try:
        Engine(dry_run="--dry-run" in sys.argv).run()
    except KeyboardInterrupt:
        log("parado por el usuario")