"""train_model.py · entrena el re-ranker LightGBM (team_40) y lo guarda en models/ranker.pkl.

    python train_model.py                      # histórico (últimos 7 días) + nuestro feedback en vivo
    python train_model.py --impressions 80000  # más rápido / menos RAM
    python train_model.py --no-feedback        # solo histórico

Metodología (para el jurado: "¿cómo se validó offline antes de salir en vivo?")
  · Point-in-time: las features agregadas (popularidad, co-views, CTR por producto, historial de
    usuario) se calculan SOLO con datos anteriores a CUT = último día − 7. Las etiquetas salen de
    las impresiones posteriores a CUT. No hay fuga del futuro.
  · Mismo código que producción: candidatos y features salen de features_cache.FeatureCache.
  · Sesgo de posición: `slot` entra como feature al entrenar y se fija a 0 al servir.
  · Sesgo de popularidad del log: cada fila pesa 1/propensity (recortado), como sugiere el README.
  · Objetivo: P(compra) = P(clic) × P(compra | clic) (dos modelos binarios; el segundo aprovecha
    mejor los pocos positivos de compra). Si el segundo no mejora en validación, se usa solo el clic.
  · Validación temporal: el último 20 % de las impresiones. Se informa AUC, MRR del producto
    clicado/comprado entre los 10 mostrados, y hit@10 del pipeline completo frente al baseline.
  · El feedback propio (team_40.feedback ⨝ shop.requests) se añade con más peso: es el simulador real.

El recomendador recarga el modelo en caliente cuando cambia el fichero: se puede reentrenar cada
pocas horas sin parar producción.
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
FEEDBACK_WEIGHT = 3.0

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
WHERE request_id != '__init__' AND {{chunk}}"""
SQL_REQUESTS = """
SELECT request_id, ts, user_id, is_new_user, context, recent_views
FROM shop.requests
WHERE {chunk}"""

SQL_HOURLY_VIEWS = """
SELECT product_id::String AS product_id, toStartOfHour(ts::DateTime64(3)) AS hour, count() AS views
FROM shop.events
WHERE event_type = 'view'
GROUP BY product_id, hour"""


# ----------------------------------------------------------------------------------------------
# Descarga
# ----------------------------------------------------------------------------------------------
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
        except Exception as e:  # noqa: BLE001
            if i > 0:
                raise
            log(f"  troceado no disponible ({str(e)[:140]}): muestra aleatoria")
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
    """team_NN.feedback ⨝ shop.requests por request_id (unidos en pandas, troceando ambos lados igual)."""
    chunks = 8
    out: list[dict] = []
    for i in range(chunks):
        cond = f"cityHash64(request_id::String) % {chunks} = {i}"
        try:
            fb = fc.query(SQL_FEEDBACK.replace("{chunk}", cond), timeout=300, name=f"feedback[{i}/{chunks}]")
            rq = fc.query(SQL_REQUESTS.replace("{chunk}", cond), timeout=300, name=f"requests[{i}/{chunks}]")
        except Exception as e:  # noqa: BLE001
            if i > 0:
                raise
            log(f"  troceado no disponible ({str(e)[:140]}): descarga completa")
            fb = fc.query(SQL_FEEDBACK.replace("{chunk}", "1 = 1"), timeout=600, name="feedback[fallback]")
            rq = fc.query(SQL_REQUESTS.replace("{chunk}", "1 = 1"), timeout=600, name="requests[fallback]")
            i = chunks
        if fb and rq:
            dfb = pd.DataFrame(fb).drop_duplicates("request_id")
            drq = pd.DataFrame(rq).drop_duplicates("request_id")
            dfb["request_id"] = dfb["request_id"].astype(str)
            drq["request_id"] = drq["request_id"].astype(str)
            out.extend(dfb.merge(drq, on="request_id", how="inner").to_dict("records"))
        log(f"  feedback con contexto: {len(out)}")
        if len(out) >= max_n or i >= chunks:
            break
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


# ----------------------------------------------------------------------------------------------
# Matriz de entrenamiento
# ----------------------------------------------------------------------------------------------
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
    """Una fila por (impresión, producto mostrado). Las features 'recientes' son point-in-time:
    histórico → vistas del día anterior; feedback → vistas de las 2 horas anteriores."""
    for r in recs:
        r["_ts"] = fc.parse_ts(r.get("ts")) or 0.0
    recs = sorted(recs, key=lambda r: r["_ts"])
    N = len(recs)
    X = np.zeros((N * 10, NF), np.float32)
    yc = np.zeros(N * 10, np.int8)
    yb = np.zeros(N * 10, np.int8)
    w = np.ones(N * 10, np.float32)
    imp = np.zeros(N * 10, np.int32)
    ts_row = np.zeros(N * 10, np.float64)
    kept: list[dict] = []
    pos, n_imp, cur_key = 0, 0, None
    days = sorted(cache.daily_views)
    for r in recs:
        ts = r["_ts"]
        if source == "history":
            key = time.strftime("%Y-%m-%d", time.gmtime(ts - 86400))
            if key != cur_key:
                cur_key = key
                counts = cache.daily_views.get(key)
                if counts is None and days:  # día sin datos: el último disponible
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
            continue  # p.ej. productos nuevos que la caché de entrenamiento no conoce
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
        ts_row[pos: pos + k] = ts
        kept.append(r)
        pos += k
        n_imp += 1
    cache.set_recent_counts(None)
    return {"X": X[:pos], "yc": yc[:pos], "yb": yb[:pos], "w": w[:pos], "imp": imp[:pos], "ts": ts_row[:pos],
            "recs": kept, "n_imp": n_imp}


def time_split(d: dict, frac_valid: float = 0.2) -> np.ndarray:
    """Máscara de validación: el último 20 % de las impresiones (están ordenadas por tiempo)."""
    return d["imp"] >= int(d["n_imp"] * (1 - frac_valid))


# ----------------------------------------------------------------------------------------------
# Entrenamiento y métricas
# ----------------------------------------------------------------------------------------------
def fit(X, y, w, Xv, yv, wv, params: dict, rounds: int = 2000, fixed_rounds: int | None = None):
    import lightgbm as lgb

    dtr = lgb.Dataset(X, y, weight=w, feature_name=FEATURES, categorical_feature=CATEGORICAL, free_raw_data=False)
    if fixed_rounds:
        return lgb.train(params, dtr, num_boost_round=fixed_rounds)
    dva = lgb.Dataset(Xv, yv, weight=wv, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(200)])


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def mrr(score: np.ndarray, label: np.ndarray, imp: np.ndarray) -> tuple[float, int]:
    """MRR del producto positivo entre los mostrados en su impresión (solo impresiones con positivo)."""
    pos_rows = np.flatnonzero(label > 0)
    if not len(pos_rows):
        return float("nan"), 0
    order = np.lexsort((-score, imp))
    rank = np.empty(len(score), np.int32)
    imp_sorted = imp[order]
    start = np.r_[0, np.flatnonzero(np.diff(imp_sorted)) + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(score)]))
    rank[order] = np.arange(len(score)) - first + 1
    return float((1.0 / rank[pos_rows]).mean()), int(len(pos_rows))


def policy_eval(cache: FeatureCache, bundle: dict, recs: list[dict], n: int) -> dict:
    """Pipeline completo (retrieval + ranking) frente al baseline en impresiones de validación con clic."""
    from recommender_v2 import Ranker, rank_batch

    ranker = Ranker(bundle)
    clicked = [r for r in recs if fc.as_pid(r.get("clicked"))][-n:]
    if not clicked:
        return {}
    base10 = set(cache.best[:10])
    hits = {"baseline": 0, "retrieval": 0, "v2_heuristico": 0, "v2_modelo": 0}
    buy_hits = {"baseline": 0, "v2_modelo": 0}
    n_buy = 0
    ips_num = ips_den = 0.0
    matches = 0
    for r in clicked:
        c = fc.as_pid(r.get("clicked"))
        b = fc.as_pid(r.get("purchased"))
        R = cache.parse_request(r)
        S = cache.session(R)
        cands = cache.candidates(R, S)
        hits["retrieval"] += c in {cache.pids[i] for i in cands.tolist()}
        hits["baseline"] += c in base10
        heur = rank_batch(cache, None, [r])[0][0] or []
        top = rank_batch(cache, ranker, [r])[0][0] or []
        hits["v2_heuristico"] += c in heur
        hits["v2_modelo"] += c in top
        if b:
            n_buy += 1
            buy_hits["baseline"] += b in base10
            buy_hits["v2_modelo"] += b in top
        shown = [str(p) for p in fc.as_list(r.get("shown"))]
        prop = fc.as_list(r.get("propensity"))
        if top and shown and prop and top[0] == shown[0]:   # SNIPS del CTR en el slot 1 (como el notebook)
            wt = 1.0 / max(fc.as_float(prop[0], 1.0), 1e-3)
            ips_den += wt
            ips_num += wt * (c == shown[0])
            matches += 1
    m = len(clicked)
    return {"impresiones_con_clic": m,
            "hit@10_clic": {k: round(v / m, 4) for k, v in hits.items()},
            "hit@10_compra": {k: round(v / max(n_buy, 1), 4) for k, v in buy_hits.items()}, "compras": n_buy,
            "snips_slot1_entre_clicadas": round(ips_num / ips_den, 4) if ips_den else None, "slot1_matches": matches}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impressions", type=int, default=120000, help="impresiones históricas (×10 filas)")
    ap.add_argument("--feedback", type=int, default=80000, help="máximo de impresiones de nuestro feedback")
    ap.add_argument("--days", type=int, default=7, help="días finales del histórico reservados para etiquetas")
    ap.add_argument("--cut", type=str, default=None, help="fecha de corte YYYY-MM-DD (por defecto: último día − days)")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--no-ips", action="store_true", help="no ponderar por 1/propensity")
    ap.add_argument("--eval-n", type=int, default=3000, help="impresiones para la evaluación del pipeline completo")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    cut = args.cut or get_cut(args.days)
    log(f"corte temporal: features < {cut} <= etiquetas")

    # 1) caché point-in-time
    cache_path = fc.CACHE_DIR / f"train_cache_{cut}.pkl"
    cache = None
    if cache_path.exists() and not args.rebuild_cache:
        try:
            cache = FeatureCache.load(cache_path)
            log(f"caché de entrenamiento: uso {cache_path.name}")
        except Exception as e:  # noqa: BLE001
            log(f"caché de entrenamiento no válida ({e})")
    if cache is None:
        log("construyendo la caché de entrenamiento (histórico anterior al corte)…")
        cache = FeatureCache.build(until=cut, with_daily=True)
        cache.save(cache_path)

    # 2) datos etiquetados
    hist = download_sessions(cut, args.impressions)
    fb: list[dict] = []
    hourly = None
    if not args.no_feedback and args.feedback > 0:
        try:
            fb = download_feedback(args.feedback)
            hourly = hourly_live_views(cache)
        except Exception as e:  # noqa: BLE001
            log(f"sin feedback en este entrenamiento: {str(e)[:200]}")
    log(f"impresiones: {len(hist)} históricas + {len(fb)} de feedback")
    if not hist and not fb:
        raise SystemExit("no hay datos de entrenamiento")

    vocab = build_vocab(hist + fb)
    parts = []
    if hist:
        parts.append(build_matrix(cache, hist, vocab, "history", use_ips=not args.no_ips))
    if fb:
        parts.append(build_matrix(cache, fb, vocab, "feedback", hourly=hourly))
    offs = np.cumsum([0] + [p["n_imp"] for p in parts])
    X = np.vstack([p["X"] for p in parts])
    yc = np.concatenate([p["yc"] for p in parts])
    yb = np.concatenate([p["yb"] for p in parts])
    w = np.concatenate([p["w"] for p in parts])
    imp = np.concatenate([p["imp"] + o for p, o in zip(parts, offs)])
    va = np.concatenate([time_split(p) for p in parts])
    tr = ~va
    del parts[1:]
    log(f"filas: {len(X)} · clics {int(yc.sum())} ({100 * yc.mean():.2f}%) · compras {int(yb.sum())} "
        f"({100 * yb.mean():.3f}%) · RAM de X: {X.nbytes / 1e6:.0f} MB")

    # 3) modelos
    threads = max(1, (os.cpu_count() or 2) - 1)
    p_click = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   cat_smooth=20, num_threads=threads, verbose=-1, seed=0)
    p_buy = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                 cat_smooth=20, num_threads=threads, verbose=-1, seed=0)
    log("entrenando P(clic)…")
    m_click = fit(X[tr], yc[tr], w[tr], X[va], yc[va], w[va], p_click)
    it_click = m_click.best_iteration or m_click.current_iteration()

    m_buy, it_buy = None, 0
    ctr, cva = tr & (yc == 1), va & (yc == 1)
    if int(yb[ctr].sum()) >= 150 and int((1 - yb[ctr]).sum()) >= 150 and int(yb[cva].sum()) >= 30:
        log("entrenando P(compra | clic)…")
        m_buy = fit(X[ctr], yb[ctr], w[ctr], X[cva], yb[cva], w[cva], p_buy)
        it_buy = m_buy.best_iteration or m_buy.current_iteration()
    else:
        log("pocos clics/compras para P(compra | clic): se usará solo P(clic)")

    # 4) validación (slot neutralizado, como en producción)
    Xv = X[va].copy()
    Xv[:, F["slot"]] = 0
    pc = m_click.predict(Xv, num_iteration=it_click)
    pb = m_buy.predict(Xv, num_iteration=it_buy) if m_buy is not None else None
    yv_c, yv_b, imp_v = yc[va], yb[va], imp[va]
    slot_score = -X[va][:, F["slot"]]
    rng = np.random.default_rng(0)
    metrics = {
        "auc_clic": auc(yv_c, pc), "auc_compra_con_p_clic": auc(yv_b, pc),
        "mrr_clic": {"modelo": mrr(pc, yv_c, imp_v)[0], "orden_sistema_antiguo": mrr(slot_score, yv_c, imp_v)[0],
                     "aleatorio": mrr(rng.random(len(pc)), yv_c, imp_v)[0]},
        "mrr_compra": {"p_clic": mrr(pc, yv_b, imp_v)[0], "orden_sistema_antiguo": mrr(slot_score, yv_b, imp_v)[0]},
        "compras_validacion": int(yv_b.sum()), "rondas_clic": int(it_click), "rondas_compra": int(it_buy),
    }
    combine = "click"
    if pb is not None:
        metrics["auc_compra_clic_x_compra"] = auc(yv_b, pc * pb)
        metrics["mrr_compra"]["p_clic_x_p_compra"] = mrr(pc * pb, yv_b, imp_v)[0]
        if metrics["mrr_compra"]["p_clic_x_p_compra"] >= metrics["mrr_compra"]["p_clic"]:
            combine = "click_x_buy"
    metrics["combinacion_elegida"] = combine
    gain = m_click.feature_importance("gain")
    metrics["importancia_gain_top"] = {FEATURES[i]: round(float(gain[i] / max(gain.sum(), 1e-9)), 4)
                                       for i in np.argsort(-gain)[:15]}

    version = "lgb-" + time.strftime("%m%dT%H%M")
    bundle = {"click": m_click.model_to_string(num_iteration=it_click),
              "buy": m_buy.model_to_string(num_iteration=it_buy) if m_buy is not None else None,
              "combine": combine, "features": list(FEATURES), "vocab": vocab, "cat_names": cache.cat_names,
              "version": version, "cut": cut, "trained_at": time.time()}
    if hist and args.eval_n > 0:
        try:
            n_valid = int(len(hist) * 0.2)
            valid_recs = sorted(hist, key=lambda r: r["_ts"])[-n_valid:]
            metrics["pipeline_completo"] = policy_eval(cache, bundle, valid_recs, args.eval_n)
        except Exception as e:  # noqa: BLE001
            log(f"evaluación del pipeline omitida: {str(e)[:200]}")
    log("validación:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))

    # 5) reentreno con todos los datos y guardado atómico (el recomendador lo recarga en caliente)
    log("reentrenando con todos los datos…")
    f_click = fit(X, yc, w, None, None, None, p_click, fixed_rounds=int(it_click * 1.1) + 1)
    bundle["click"] = f_click.model_to_string()
    if m_buy is not None:
        allc = yc == 1
        f_buy = fit(X[allc], yb[allc], w[allc], None, None, None, p_buy, fixed_rounds=int(it_buy * 1.1) + 1)
        bundle["buy"] = f_buy.model_to_string()
    bundle["metrics"] = metrics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_DIR / "ranker.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, MODEL_DIR / "ranker.pkl")
    (MODEL_DIR / f"metrics_{version}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    log(f"modelo {version} guardado en {MODEL_DIR / 'ranker.pkl'} · total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
