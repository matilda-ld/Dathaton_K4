"""train_model.py · Entrena el ensemble LightGBM calibrado sin distorsión de precios (team_40).

    python train_model.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import features_cache as fc
from features_cache import CATEGORICAL, F, FEATURES, NF, FeatureCache, log

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", HERE / "models"))
IPS_CLIP = 5.0
FEEDBACK_WEIGHT = 4.5   # Nuestro propio feedback tiene máxima prioridad porque viene del simulador en vivo

SQL_LAST_DAY = "SELECT toDate(max(ts::DateTime64(3))) AS d FROM shop.history_sessions"

SQL_SESSIONS = """
SELECT request_id, ts, user_id, context, recent_views, shown, propensity, clicked, purchased
FROM shop.history_sessions
WHERE ts >= '{cut}' AND {chunk}"""
SQL_SESSIONS_FALLBACK = """
SELECT request_id, ts, user_id, context, recent_views, shown, propensity, clicked, purchased
FROM shop.history_sessions WHERE ts >= '{cut}' ORDER BY rand() LIMIT {n}"""

SQL_FEEDBACK = f"""
SELECT request_id, shown, served_by, clicked, purchased
FROM {fc.TEAM}.feedback
WHERE request_id != '__init__' AND served_by = 'team' AND {{chunk}}"""
SQL_REQUESTS = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
WHERE {chunk}"""

SQL_HOURLY_VIEWS = """
SELECT product_id::String AS product_id, toStartOfHour(ts::DateTime64(3)) AS hour, count() AS views
FROM shop.events
WHERE event_type = 'view'
GROUP BY product_id, hour"""


def get_cut(days: int) -> str:
    d = str(fc.query(SQL_LAST_DAY, timeout=120, name="last_day")[0]["d"])[:10]
    return str((pd.Timestamp(d) - pd.Timedelta(days=days)).date())


def download_sessions(cut: str, n: int) -> list[dict]:
    path = fc.CACHE_DIR / f"sessions_{cut}_{n}.pkl"
    if path.exists():
        log(f"impresiones históricas: uso {path.name}")
        return pd.read_pickle(path).to_dict("records")
    chunks = 16
    rows: list[dict] = []
    tpl = SQL_SESSIONS.replace("{cut}", cut)
    for i in range(chunks):
        try:
            part = fc.query(tpl.replace("{chunk}", f"cityHash64(request_id::String) % {chunks} = {i}"),
                            timeout=600, name=f"sessions[{i}/{chunks}]")
        except Exception:  # noqa: BLE001
            if i > 0:
                raise
            part = fc.query(SQL_SESSIONS_FALLBACK.replace("{cut}", cut).replace("{n}", str(min(n, 100000))),
                            timeout=600, name="sessions[fallback]")
            rows.extend(part)
            break
        rows.extend(part)
        log(f"  impresiones históricas: {len(rows)}")
        if len(rows) >= n:
            break
    rows = rows[:n]
    fc.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(path)
    return rows


def download_feedback(max_n: int) -> list[dict]:
    chunks = 8
    out: list[dict] = []
    for i in range(chunks):
        cond = f"cityHash64(request_id::String) % {chunks} = {i}"
        try:
            fb = fc.query(SQL_FEEDBACK.replace("{chunk}", cond), timeout=300, name=f"feedback[{i}/{chunks}]")
            rq = fc.query(SQL_REQUESTS.replace("{chunk}", cond), timeout=300, name=f"requests[{i}/{chunks}]")
        except Exception:  # noqa: BLE001
            if i > 0:
                raise
            fb = fc.query(SQL_FEEDBACK.replace("{chunk}", "1 = 1"), timeout=600, name="feedback[fallback]")
            rq = fc.query(SQL_REQUESTS.replace("{chunk}", "1 = 1"), timeout=600, name="requests[fallback]")
            i = chunks
        if fb and rq:
            dfb = pd.DataFrame(fb).drop_duplicates("request_id")
            drq = pd.DataFrame(rq).drop_duplicates("request_id")
            dfb["request_id"] = dfb["request_id"].astype(str)
            drq["request_id"] = drq["request_id"].astype(str)
            out.extend(dfb.merge(drq, on="request_id", how="inner").to_dict("records"))
        if len(out) >= max_n or i >= chunks:
            break
    log(f"  feedback propio descargado: {len(out)} impresiones")
    return out[:max_n]


def hourly_live_views(cache: FeatureCache) -> dict[int, np.ndarray]:
    rows = fc.query(SQL_HOURLY_VIEWS, timeout=300, name="hourly_views")
    out: dict[int, np.ndarray] = {}
    for r in rows:
        idx = cache.pid2idx.get(str(r["product_id"]))
        h = fc.parse_ts(r.get("hour"))
        if idx is None or h is None:
            continue
        key = int(h // 3600)
        arr = out.get(key)
        if arr is None:
            arr = out[key] = np.zeros(cache.cap, np.float32)
        arr[idx] = fc.as_float(r.get("views"), 0)
    return out


def build_vocab(recs: list[dict]) -> dict:
    vals: dict[str, set] = {"device": set(), "region": set(), "page": set()}
    for r in recs:
        ctx = fc.as_obj(r.get("context"))
        for k in vals:
            if ctx.get(k) is not None:
                vals[k].add(str(ctx[k]))
    return {k: {v: i for i, v in enumerate(sorted(s))} for k, s in vals.items()}


def build_matrix(cache: FeatureCache, recs: list[dict], vocab: dict, source: str,
                 hourly: dict[int, np.ndarray] | None = None, use_ips: bool = True) -> dict:
    for r in recs:
        r["_ts"] = fc.parse_ts(r.get("ts")) or 0.0
    recs = sorted(recs, key=lambda r: r["_ts"])
    N = len(recs)
    X = np.zeros((N * 10, NF), np.float32)
    yc = np.zeros(N * 10, np.int8)
    yb = np.zeros(N * 10, np.int8)
    w = np.ones(N * 10, np.float32)
    imp = np.zeros(N * 10, np.int32)
    pos, n_imp, cur_key = 0, 0, None
    days = sorted(cache.daily_views)
    for r in recs:
        ts = r["_ts"]
        if source == "history":
            key = time.strftime("%Y-%m-%d", time.gmtime(ts - 86400))
            if key != cur_key:
                cur_key = key
                counts = cache.daily_views.get(key)
                if counts is None and days:
                    prev = [d for d in days if d <= key]
                    counts = cache.daily_views[prev[-1] if prev else days[0]]
                cache.set_recent_counts(counts)
        else:
            key = int(ts // 3600)
            if key != cur_key:
                cur_key = key
                counts = None
                if hourly:
                    a, b = hourly.get(key - 1), hourly.get(key - 2)
                    counts = a + b if a is not None and b is not None else (a if a is not None else b)
                if counts is None and days:
                    counts = cache.daily_views[days[-1]]
                cache.set_recent_counts(counts)
        shown = [cache.pid2idx.get(str(p)) for p in fc.as_list(r.get("shown"))]
        if not shown or any(i is None for i in shown):
            continue
        k = len(shown)
        if pos + k > len(X):
            break
        R = cache.parse_request(r)
        S = cache.session(R)
        cands = np.asarray(shown, np.int64)
        X[pos: pos + k] = cache.featurize(R, S, cands, vocab, slots=np.arange(k, dtype=np.float32))
        c = cache.pid2idx.get(fc.as_pid(r.get("clicked")), -1)
        b = cache.pid2idx.get(fc.as_pid(r.get("purchased")), -1)
        yc[pos: pos + k] = (cands == c) | (cands == b)
        yb[pos: pos + k] = cands == b
        if source == "history" and use_ips:
            prop = [fc.as_float(p, 1.0) for p in fc.as_list(r.get("propensity"))]
            if len(prop) == k:
                w[pos: pos + k] = np.clip(1.0 / np.maximum(np.asarray(prop, np.float32), 1e-3), 1.0, IPS_CLIP)
        elif source == "feedback":
            w[pos: pos + k] = FEEDBACK_WEIGHT
        imp[pos: pos + k] = n_imp
        pos += k
        n_imp += 1
    cache.set_recent_counts(None)
    return {"X": X[:pos], "yc": yc[:pos], "yb": yb[:pos], "w": w[:pos], "imp": imp[:pos], "n_imp": n_imp}


def time_split(d: dict, frac_valid: float = 0.2) -> np.ndarray:
    return d["imp"] >= int(d["n_imp"] * (1 - frac_valid))


def fit(X, y, w, Xv, yv, wv, params: dict, rounds: int = 2000, fixed_rounds: int | None = None):
    import lightgbm as lgb

    dtr = lgb.Dataset(X, y, weight=w, feature_name=FEATURES, categorical_feature=CATEGORICAL, free_raw_data=False)
    if fixed_rounds:
        return lgb.train(params, dtr, num_boost_round=fixed_rounds)
    dva = lgb.Dataset(Xv, yv, weight=wv, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(200)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impressions", type=int, default=120000)
    ap.add_argument("--feedback", type=int, default=90000)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--cut", type=str, default=None)
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    cut = args.cut or get_cut(args.days)
    log(f"corte temporal: features < {cut} <= etiquetas")

    cache_path = fc.CACHE_DIR / f"train_cache_{cut}.pkl"
    cache = None
    if cache_path.exists() and not args.rebuild_cache:
        try:
            cache = FeatureCache.load(cache_path)
            log(f"caché de entrenamiento: {cache_path.name}")
        except Exception:  # noqa: BLE001
            cache = None
    if cache is None:
        cache = FeatureCache.build(until=cut, with_daily=True)
        cache.save(cache_path)

    # Actualizar catálogo de la caché de entrenamiento para que también conozca los 200 productos nuevos del evento
    try:
        cache.refresh_catalog()
    except Exception:  # noqa: BLE001
        pass

    hist = download_sessions(cut, args.impressions)
    fb: list[dict] = []
    hourly = None
    if not args.no_feedback and args.feedback > 0:
        try:
            fb = download_feedback(args.feedback)
            hourly = hourly_live_views(cache)
        except Exception as e:  # noqa: BLE001
            log(f"sin feedback: {str(e)[:150]}")

    vocab = build_vocab(hist + fb)
    parts = []
    if hist:
        parts.append(build_matrix(cache, hist, vocab, "history", use_ips=True))
    if fb:
        parts.append(build_matrix(cache, fb, vocab, "feedback", hourly=hourly))
    X = np.vstack([p["X"] for p in parts])
    yc = np.concatenate([p["yc"] for p in parts])
    yb = np.concatenate([p["yb"] for p in parts])
    w = np.concatenate([p["w"] for p in parts])
    va = np.concatenate([time_split(p) for p in parts])
    tr = ~va

    threads = max(1, (os.cpu_count() or 2) - 1)
    p_click = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.5,
                   cat_smooth=20, num_threads=threads, verbose=-1, seed=0)
    p_buy = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                 cat_smooth=20, num_threads=threads, verbose=-1, seed=0)

    log("1/3 entrenando P(clic)…")
    m_click = fit(X[tr], yc[tr], w[tr], X[va], yc[va], w[va], p_click)
    it_click = m_click.best_iteration or m_click.current_iteration()

    log("2/3 entrenando P(compra directa)…")
    m_dir = fit(X[tr], yb[tr], w[tr], X[va], yb[va], w[va], p_buy)
    it_dir = m_dir.best_iteration or m_dir.current_iteration()

    m_buy, it_buy = None, 0
    ctr, cva = tr & (yc == 1), va & (yc == 1)
    if int(yb[ctr].sum()) >= 100 and int(yb[cva].sum()) >= 20:
        log("3/3 entrenando P(compra | clic)…")
        m_buy = fit(X[ctr], yb[ctr], w[ctr], X[cva], yb[cva], w[cva], p_buy)
        it_buy = m_buy.best_iteration or m_buy.current_iteration()

    log("reentrenando sobre el 100% de los datos…")
    f_click = fit(X, yc, w, None, None, None, p_click, fixed_rounds=int(it_click * 1.1) + 1)
    f_dir = fit(X, yb, w, None, None, None, p_buy, fixed_rounds=int(it_dir * 1.1) + 1)
    f_buy = None
    if m_buy is not None:
        allc = yc == 1
        f_buy = fit(X[allc], yb[allc], w[allc], None, None, None, p_buy, fixed_rounds=int(it_buy * 1.1) + 1)

    version = "lgb-" + time.strftime("%m%dT%H%M")
    bundle = {
        "click": f_click.model_to_string(),
        "buy": f_buy.model_to_string() if f_buy is not None else None,
        "direct_buy": f_dir.model_to_string(),
        "combine": "ensemble",
        "features": list(FEATURES),
        "vocab": vocab,
        "cat_names": cache.cat_names,
        "version": version,
        "cut": cut,
        "trained_at": time.time(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_DIR / "ranker.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, MODEL_DIR / "ranker.pkl")
    log(f"modelo {version} guardado en {MODEL_DIR / 'ranker.pkl'} en {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()