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
import threading
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


def session_chunks(cut: str, n: int):
    """Impresiones históricas posteriores al corte, de trozo en trozo (cada uno se cachea en disco y se
    procesa por separado: la RAM no crece con el número de impresiones)."""
    chunks = 16
    total = 0
    tpl = SQL_SESSIONS.replace("{cut}", cut)
    fc.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(chunks):
        path = fc.CACHE_DIR / f"sessions_{cut}_c{i:02d}.pkl"
        if path.exists():
            part = pd.read_pickle(path).to_dict("records")
        else:
            try:
                part = fc.query(tpl.replace("{chunk}", f"cityHash64(request_id::String) % {chunks} = {i}"),
                                timeout=600, name=f"sessions[{i}/{chunks}]")
            except Exception as e:  # noqa: BLE001
                if i > 0:
                    raise
                log(f"  troceado no disponible ({str(e)[:140]}): muestra aleatoria")
                yield fc.query(SQL_SESSIONS_FALLBACK.replace("{cut}", cut).replace("{n}", str(min(n, 100000))),
                               timeout=600, name="sessions[fallback]")
                return
            pd.DataFrame(part).to_pickle(path)
        total += len(part)
        yield part
        if total >= n:
            return


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
                 hourly: dict[int, np.ndarray] | None = None, use_ips: bool = True, neg_keep: int = 3,
                 seed: int = 0) -> dict:
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
    rng = np.random.default_rng(seed)
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
        c = cache.pid2idx.get(fc.as_pid(r.get("clicked")), -1)
        b = cache.pid2idx.get(fc.as_pid(r.get("purchased")), -1)
        lab_c = (cands == c) | (cands == b)
        wr = np.ones(k, np.float32)
        if source == "history" and use_ips:
            prop = [fc.as_float(p, 1.0) for p in fc.as_list(r.get("propensity"))]
            if len(prop) == k:
                wr = np.clip(1.0 / np.maximum(np.asarray(prop, np.float32), 1e-3), 1.0, IPS_CLIP)
        elif source == "feedback":
            wr *= FEEDBACK_WEIGHT
        slots = np.arange(k, dtype=np.float32)
        if source == "history" and S.bought.size:
            # El histórico se generó SIN la regla "lo ya comprado casi no se recompra" (aclaración de la
            # organización). Esas filas enseñarían al modelo justo lo contrario de lo que puntúa: fuera.
            # El feedback propio sí sigue la regla y se queda entero.
            keep = ~np.isin(cands, S.bought)
            if not keep.any():
                continue
            cands, lab_c, slots, wr = cands[keep], lab_c[keep], slots[keep], wr[keep]
            k = len(cands)
        if not lab_c.any() and 0 < neg_keep < k:      # impresión sin clic: basta una muestra de sus filas
            keep = np.sort(rng.choice(k, neg_keep, replace=False))
            cands, lab_c, slots, wr = cands[keep], lab_c[keep], slots[keep], wr[keep] * (k / neg_keep)
            k = neg_keep
        X[pos: pos + k] = cache.featurize(R, S, cands, vocab, slots=slots)
        yc[pos: pos + k] = lab_c
        yb[pos: pos + k] = cands == b
        w[pos: pos + k] = wr
        imp[pos: pos + k] = n_imp
        ts_row[pos: pos + k] = ts
        kept.append(r)
        pos += k
        n_imp += 1
    cache.set_recent_counts(None)
    if source == "feedback" and pos:
        # El feedback más reciente refleja mejor la política y las tendencias actuales: semivida de 6 h,
        # de ~2× el peso base (recién llegado) a ~0,5× (más de un día).
        age_h = (ts_row[:pos].max() - ts_row[:pos]) / 3600.0
        w[:pos] *= (0.5 + 1.5 * 0.5 ** (age_h / 6.0)).astype(np.float32)
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


def group_sizes(imp_rows: np.ndarray) -> np.ndarray:
    """Tamaño de cada impresión (sus filas van seguidas) para el objetivo de ranking."""
    cuts = np.flatnonzero(np.r_[True, imp_rows[1:] != imp_rows[:-1], True])
    return np.diff(cuts)


def fit_rank(X, y, w, imp_rows, Xv, yv, wv, imp_v, threads: int, fixed_rounds: int | None = None):
    """LambdaMART: en vez de puntuar cada fila por separado, aprende a ORDENAR los productos de una misma
    impresión (compra = 2, clic sin compra = 1, ignorado = 0). Solo aportan las impresiones con algún clic."""
    import lightgbm as lgb

    params = dict(objective="lambdarank", metric="ndcg", eval_at=[10], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  lambdarank_truncation_level=10, cat_smooth=20, num_threads=threads, verbose=-1, seed=0)
    dtr = lgb.Dataset(X, y, weight=w, group=group_sizes(imp_rows), feature_name=FEATURES,
                      categorical_feature=CATEGORICAL, free_raw_data=False)
    if fixed_rounds:
        return lgb.train(params, dtr, num_boost_round=fixed_rounds)
    dva = lgb.Dataset(Xv, yv, weight=wv, group=group_sizes(imp_v), reference=dtr)
    return lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(200)])


def auc(y, p, w=None) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p, sample_weight=w)) if 0 < y.sum() < len(y) else float("nan")


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


def cart_boost_check(Xv, score, yv_b, imp_v, mask) -> dict:
    """MRR de la compra con y sin subir el carrito, en las impresiones (de feedback) que tienen carrito."""
    from recommender_v2 import apply_cart_boost

    rows = np.flatnonzero(mask)
    has_cart = np.zeros(len(score), bool)
    has_cart[rows[Xv[rows, F["lc_in"]] > 0]] = True
    imps = np.unique(imp_v[has_cart])
    m = np.isin(imp_v, imps) & mask
    n_buy = int(yv_b[m].sum())
    if n_buy < 20:
        return {"activar": False, "motivo": f"pocas compras en impresiones con carrito ({n_buy})"}
    boosted = score.copy()
    idx = np.flatnonzero(m)
    order = np.argsort(imp_v[idx], kind="stable")
    idx = idx[order]
    cuts = np.flatnonzero(np.r_[True, imp_v[idx][1:] != imp_v[idx][:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        g = idx[a:b]
        boosted[g] = apply_cart_boost(score[g], Xv[g])
    base_mrr = mrr(score[m], yv_b[m], imp_v[m])[0]
    new_mrr = mrr(boosted[m], yv_b[m], imp_v[m])[0]
    return {"activar": bool(new_mrr > base_mrr), "mrr_sin": round(base_mrr, 4), "mrr_con": round(new_mrr, 4),
            "compras": n_buy, "impresiones": int(len(imps))}


def champion_check(Xv, new_score, yv_b, imp_v, fb_mask, ts_v) -> dict:
    """Compara el aspirante con el modelo de models/ranker.pkl en las impresiones de feedback posteriores a
    cuando se entrenó ese modelo (ninguno de los dos las ha visto: el aspirante deja fuera la validación)."""
    path = MODEL_DIR / "ranker.pkl"
    if not path.exists():
        return {"desplegar": True, "motivo": "no hay modelo en producción"}
    try:
        from recommender_v2 import Ranker

        with open(path, "rb") as f:
            old_bundle = pickle.load(f)
        old = Ranker(old_bundle)
        since = float(old_bundle.get("trained_at", 0.0))
    except Exception as e:  # noqa: BLE001
        return {"desplegar": True, "motivo": f"el modelo actual no se puede usar ({str(e)[:80]})"}
    win = fb_mask & (ts_v > since)
    n_buy = int(yv_b[win].sum())
    if n_buy < 40:
        return {"desplegar": True, "motivo": f"pocas compras nuevas para comparar ({n_buy})", "actual": old_bundle.get("version")}
    old_score = old.score(Xv[win])
    m_new = mrr(new_score[win], yv_b[win], imp_v[win])[0]
    m_old = mrr(old_score, yv_b[win], imp_v[win])[0]
    return {"desplegar": bool(m_new >= m_old), "mrr_compra_nuevo": round(m_new, 4), "mrr_compra_actual": round(m_old, 4),
            "compras_comparadas": n_buy, "actual": old_bundle.get("version")}


def policy_eval(cache: FeatureCache, bundle: dict, recs: list[dict], n: int) -> dict:
    """Pipeline completo frente al baseline en impresiones de validación con clic, con los dos brazos:
    'cand' (retrieval + re-ranking) y 'full' (el modelo puntúa todo el catálogo)."""
    from recommender_v2 import Ranker, rank_batch

    ranker = Ranker(bundle)
    def fresh_click(r: dict) -> bool:      # solo clics en productos que el usuario NO había comprado ya
        c = cache.pid2idx.get(fc.as_pid(r.get("clicked")))
        if c is None:
            return False
        R = cache.parse_request(r)
        return c not in set(cache.session(R).bought.tolist())

    clicked = [r for r in recs if fresh_click(r)][-n:]
    if not clicked:
        return {}
    base10 = set(cache.best[:10])
    hits = {"baseline": 0, "retrieval": 0, "cand": 0, "wide": 0}
    buy_hits = {"baseline": 0, "cand": 0, "wide": 0}
    n_buy = 0
    for i in range(0, len(clicked), 4):
        chunk = clicked[i: i + 4]
        tops = {arm: rank_batch(cache, ranker, chunk, [arm] * len(chunk)) for arm in ("cand", "wide")}
        for j, r in enumerate(chunk):
            c, b = fc.as_pid(r.get("clicked")), fc.as_pid(r.get("purchased"))
            R = cache.parse_request(r)
            cands = cache.candidates(R, cache.session(R))
            hits["retrieval"] += c in {cache.pids[k] for k in cands.tolist()}
            hits["baseline"] += c in base10
            n_buy += bool(b)
            buy_hits["baseline"] += bool(b) and b in base10
            for arm in ("cand", "wide"):
                top = tops[arm][j][0] or []
                hits[arm] += c in top
                buy_hits[arm] += bool(b) and b in top
    m = len(clicked)
    return {"impresiones_con_clic": m, "compras": n_buy,
            "hit@10_clic": {k: round(v / m, 4) for k, v in hits.items()},
            "hit@10_compra": {k: round(v / max(n_buy, 1), 4) for k, v in buy_hits.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impressions", type=int, default=400000, help="impresiones históricas (×10 filas)")
    ap.add_argument("--feedback", type=int, default=200000, help="máximo de impresiones de nuestro feedback")
    ap.add_argument("--days", type=int, default=14, help="días finales del histórico reservados para etiquetas")
    ap.add_argument("--cut", type=str, default=None, help="fecha de corte YYYY-MM-DD (por defecto: último día − days)")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--no-ips", action="store_true", help="no ponderar por 1/propensity")
    ap.add_argument("--eval-n", type=int, default=1500, help="impresiones para la evaluación del pipeline completo")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="MIN",
                    help="reentrenar cada MIN minutos con el feedback nuevo (el recomendador recarga en caliente)")
    args = ap.parse_args()
    final_start = fc.parse_ts(os.environ.get("FINAL_START_UTC", "2026-11-15T09:00:00Z")) or 0.0
    if args.loop:
        beat = MODEL_DIR / "trainer.heartbeat"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if beat.exists() and time.time() - beat.stat().st_mtime < 90:
            log("ya hay otro entrenamiento en bucle en marcha: este no hace falta, salgo")
            return

        def heartbeat() -> None:
            while True:
                try:
                    beat.write_text(str(os.getpid()))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(30)

        threading.Thread(target=heartbeat, daemon=True).start()
    while True:
        if args.loop and final_start and time.time() > final_start - 20 * 60 and (MODEL_DIR / "ranker.pkl").exists():
            log("faltan menos de 20 min para la final (o ya ha empezado): no se reentrena más")
            break
        wait = args.loop * 60
        try:
            run_once(args)
        except Exception as e:  # noqa: BLE001
            if not args.loop:
                raise
            log(f"entrenamiento fallido, se reintenta en 2 min: {str(e)[:300]}")
            wait = 120
        if not args.loop:
            break
        log(f"próximo entrenamiento en {wait // 60} min")
        time.sleep(wait)


def run_once(args) -> None:
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

    # 2) perfil de usuario con hora (carritos y compras): histórico desde 2 días antes del corte + stream
    cache.user_ev, cache._ev_ts = {}, 0.0
    since = str((pd.Timestamp(cut) - pd.Timedelta(days=2)).date())
    try:
        log(f"eventos de usuario de shop.history_events: {cache.load_user_events('shop.history_events', since)}")
    except Exception as e:  # noqa: BLE001
        log(f"sin eventos de usuario de shop.history_events: {str(e)[:200]}")

    # 3) datos etiquetados (el histórico se procesa trozo a trozo)
    fb: list[dict] = []
    hourly = None
    if not args.no_feedback and args.feedback > 0:
        try:
            fb = download_feedback(args.feedback)
            hourly = hourly_live_views(cache)
        except Exception as e:  # noqa: BLE001
            log(f"sin feedback en este entrenamiento: {str(e)[:200]}")
    parts, sources, valid_recs, vocab, n_hist = [], [], [], None, 0
    for ci, chunk in enumerate(session_chunks(cut, args.impressions)):
        if vocab is None:
            vocab = build_vocab(chunk + fb)
        part = build_matrix(cache, chunk, vocab, "history", use_ips=not args.no_ips, seed=ci)
        recs = part.pop("recs")
        valid_recs.extend(r for r in recs[int(len(recs) * 0.8):] if fc.as_pid(r.get("clicked")))
        n_hist += part["n_imp"]
        parts.append(part)
        sources.append("history")
        log(f"  impresiones históricas procesadas: {n_hist}")
    if vocab is None:
        vocab = build_vocab(fb)
    if fb:
        # Los productos dados de alta en directo entran en la caché igual que en producción (features
        # heredadas de sus vecinos de texto): así el modelo aprende del simulador cómo rinden de verdad.
        state = (cache.active.copy(), cache.oos_cat.copy())
        try:
            log(f"productos nuevos incorporados para el feedback: {cache.refresh_catalog()}")
        except Exception as e:  # noqa: BLE001
            log(f"no se pudo leer catalog_updates: {str(e)[:160]}")
        # Ahora que existen los productos nuevos: eventos en vivo (carritos y compras orgánicas) y las compras
        # hechas con nuestras recomendaciones, con su hora. Antes se cargaban demasiado pronto y se perdían
        # las de productos nuevos; las compras propias no se cargaban.
        for label, fn in (("eventos de usuario de shop.events", lambda: cache.load_user_events("shop.events")),
                          ("compras con nuestras recomendaciones", cache.add_own_purchases_ts)):
            try:
                log(f"{label}: {fn()}")
            except Exception as e:  # noqa: BLE001
                log(f"sin {label}: {str(e)[:200]}")
        part = build_matrix(cache, fb, vocab, "feedback", hourly=hourly, seed=99)
        part.pop("recs")
        parts.append(part)
        sources.append("feedback")
        cache.active[:], cache.oos_cat[:] = state      # el stock de hoy no debe afectar a la validación histórica
        cache.oos_fast[:] = False
        cache._refresh_avail()
    if not parts or not sum(p["n_imp"] for p in parts):
        raise RuntimeError("no hay datos de entrenamiento")
    log(f"impresiones: {n_hist} históricas + {parts[-1]['n_imp'] if fb else 0} de feedback")
    offs = np.cumsum([0] + [p["n_imp"] for p in parts])
    X = np.vstack([p["X"] for p in parts])
    yc = np.concatenate([p["yc"] for p in parts])
    yb = np.concatenate([p["yb"] for p in parts])
    w = np.concatenate([p["w"] for p in parts])
    imp = np.concatenate([p["imp"] + o for p, o in zip(parts, offs)])
    va = np.concatenate([time_split(p) for p in parts])
    is_fb = np.concatenate([np.full(len(p["yc"]), src == "feedback") for p, src in zip(parts, sources)])
    ts_all = np.concatenate([p["ts"] for p in parts])
    tr = ~va
    n_fb_used = int(parts[-1]["n_imp"]) if fb else 0
    del parts
    log(f"filas (negativos submuestreados): {len(X)} · clics {int(yc.sum())} · compras {int(yb.sum())} · "
        f"RAM de X: {X.nbytes / 1e6:.0f} MB")

    # 3) modelos
    threads = int(os.environ.get("TRAIN_THREADS", 0)) or max(1, (os.cpu_count() or 2) - 1)
    p_click = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   cat_smooth=20, num_threads=threads, verbose=-1, seed=0)
    p_buy = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                 cat_smooth=20, num_threads=threads, verbose=-1, seed=0)
    # Lo que puntúa es el simulador y nuestro feedback es su único reflejo: si hay datos suficientes, el early
    # stopping y la elección de modelo se deciden SOLO con la validación de feedback.
    fb_val = va & is_fb
    use_fb = int(yb[fb_val].sum()) >= 50 and int(yc[fb_val].sum()) >= 150
    es = fb_val if use_fb else va
    log(f"validación para decidir: {'solo feedback' if use_fb else 'histórico + feedback'} "
        f"({int(es.sum())} filas, {int(yb[es].sum())} compras)")
    log("entrenando P(clic)…")
    m_click = fit(X[tr], yc[tr], w[tr], X[es], yc[es], w[es], p_click)
    it_click = m_click.best_iteration or m_click.current_iteration()

    m_buy, it_buy = None, 0
    ctr, cva = tr & (yc == 1), es & (yc == 1)
    if int(yb[cva].sum()) < 30:
        cva = va & (yc == 1)
    if int(yb[ctr].sum()) >= 150 and int((1 - yb[ctr]).sum()) >= 150 and int(yb[cva].sum()) >= 30:
        log("entrenando P(compra | clic)…")
        m_buy = fit(X[ctr], yb[ctr], w[ctr], X[cva], yb[cva], w[cva], p_buy)
        it_buy = m_buy.best_iteration or m_buy.current_iteration()
    else:
        log("pocos clics/compras para P(compra | clic): se usará solo P(clic)")
    m_dir, it_dir = None, 0
    if int(yb[tr].sum()) >= 300 and int(yb[es].sum()) >= 50:
        log("entrenando P(compra) directa…")
        m_dir = fit(X[tr], yb[tr], w[tr], X[es], yb[es], w[es], dict(p_buy, num_leaves=31, min_data_in_leaf=80))
        it_dir = m_dir.best_iteration or m_dir.current_iteration()

    # modelo de ranking (lambdarank) como candidato adicional: solo se usará si ordena mejor la compra
    m_rank, it_rank = None, 0
    grade = (yc.astype(np.int32) + yb.astype(np.int32))
    has_pos = np.isin(imp, np.unique(imp[yc == 1]))
    rtr, rva = tr & has_pos, es & has_pos
    try:
        if int(yb[rtr].sum()) >= 300 and int(yb[rva].sum()) >= 50:
            log("entrenando el modelo de ranking (lambdarank)…")
            m_rank = fit_rank(X[rtr], grade[rtr], w[rtr], imp[rtr], X[rva], grade[rva], w[rva], imp[rva], threads)
            it_rank = m_rank.best_iteration or m_rank.current_iteration()
    except Exception as e:  # noqa: BLE001
        log(f"modelo de ranking omitido: {str(e)[:200]}")
        m_rank = None

    # 4) validación (slot neutralizado, como en producción)
    Xv = X[va].copy()
    Xv[:, F["slot"]] = 0
    pc = m_click.predict(Xv, num_iteration=it_click)
    pb = m_buy.predict(Xv, num_iteration=it_buy) if m_buy is not None else None
    yv_c, yv_b, imp_v = yc[va], yb[va], imp[va]
    slot_score = -X[va][:, F["slot"]]
    rng = np.random.default_rng(0)
    wv = w[va]
    metrics = {
        "auc_clic": auc(yv_c, pc, wv), "auc_compra_con_p_clic": auc(yv_b, pc, wv),
        "mrr_clic": {"modelo": mrr(pc, yv_c, imp_v)[0], "orden_sistema_antiguo": mrr(slot_score, yv_c, imp_v)[0],
                     "aleatorio": mrr(rng.random(len(pc)), yv_c, imp_v)[0]},
        "mrr_compra": {"orden_sistema_antiguo": mrr(slot_score, yv_b, imp_v)[0]},
        "compras_validacion": int(yv_b.sum()), "rondas_clic": int(it_click), "rondas_compra": int(it_buy),
    }
    # candidatos a puntuación final: se queda el que mejor ordena la COMPRA en validación
    options = {"click": pc}
    if pb is not None:
        options["click_x_buy"] = pc * pb
    if m_dir is not None:
        pd_ = m_dir.predict(Xv, num_iteration=it_dir)
        options["direct"] = pd_
        if pb is not None:
            options["blend"] = np.sqrt(np.maximum(pc * pb, 1e-12) * np.maximum(pd_, 1e-12))
            options["blend_add"] = 0.6 * pc * pb + 0.4 * pd_
    if m_rank is not None:
        options["rank"] = m_rank.predict(Xv, num_iteration=it_rank)
    for name, sc in options.items():
        metrics["mrr_compra"][name] = mrr(sc, yv_b, imp_v)[0]
        metrics.setdefault("auc_compra", {})[name] = auc(yv_b, sc, wv)
    sel = is_fb[va] if use_fb else np.ones(int(va.sum()), bool)
    sel_mrr = {k: mrr(v[sel], yv_b[sel], imp_v[sel])[0] for k, v in options.items()}
    metrics["mrr_compra_para_elegir"] = {"sobre": "feedback" if use_fb else "todo", **sel_mrr}
    combine = max(options, key=lambda k: (sel_mrr[k], metrics["auc_compra"][k]))
    metrics["combinacion_elegida"] = combine
    fbv = is_fb[va]
    if fbv.any():   # la parte que más importa: validación SOLO con el simulador real (nuestro feedback)
        final = options[combine]
        metrics["solo_feedback"] = {
            "impresiones_usadas": n_fb_used, "compras_validacion": int(yv_b[fbv].sum()),
            "auc_clic": auc(yv_c[fbv], pc[fbv], wv[fbv]), "auc_compra": auc(yv_b[fbv], final[fbv], wv[fbv]),
            "mrr_compra_modelo": mrr(final[fbv], yv_b[fbv], imp_v[fbv])[0],
            "mrr_compra_por_opcion": {k: mrr(v[fbv], yv_b[fbv], imp_v[fbv])[0] for k, v in options.items()},
            "mrr_compra_orden_mostrado": mrr(slot_score[fbv], yv_b[fbv], imp_v[fbv])[0]}
    gain = m_click.feature_importance("gain")
    metrics["importancia_gain_top"] = {FEATURES[i]: round(float(gain[i] / max(gain.sum(), 1e-9)), 4)
                                       for i in np.argsort(-gain)[:20]}

    version = "lgb-" + time.strftime("%m%dT%H%M")
    bundle = {"click": m_click.model_to_string(num_iteration=it_click),
              "buy": m_buy.model_to_string(num_iteration=it_buy) if m_buy is not None else None,
              "direct": m_dir.model_to_string(num_iteration=it_dir) if m_dir is not None else None,
              "rank": m_rank.model_to_string(num_iteration=it_rank) if m_rank is not None else None,
              "cache_blocks": sorted(k for k, v in cache.status.items() if v == "OK"),
              "combine": combine, "features": list(FEATURES), "vocab": vocab, "cat_names": cache.cat_names,
              "version": version, "cut": cut, "trained_at": time.time(), "wide_pool": False}
    if valid_recs and args.eval_n > 0:
        try:
            pe = metrics["pipeline_completo"] = policy_eval(cache, bundle, valid_recs, args.eval_n)
            hb, hc = pe.get("hit@10_compra", {}), pe.get("hit@10_clic", {})
            bundle["wide_pool"] = (hb.get("wide", 0), hc.get("wide", 0)) > (hb.get("cand", 0), hc.get("cand", 0))
            metrics["brazo_preferido_offline"] = "wide" if bundle["wide_pool"] else "cand"
        except Exception as e:  # noqa: BLE001
            log(f"evaluación del pipeline omitida: {str(e)[:200]}")
    log("validación:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))

    # 4b) ¿subir al principio lo que el usuario tiene en el carrito ordena mejor la compra? Se mide en el
    #     feedback de validación, solo en las impresiones con algún producto del carrito entre los mostrados.
    bundle["cart_boost"] = False
    try:
        cb = cart_boost_check(Xv, options[combine], yv_b, imp_v, sel)
        metrics["prioridad_carrito"] = cb
        bundle["cart_boost"] = bool(cb.get("activar"))
        log(f"prioridad al carrito: {cb}")
    except Exception as e:  # noqa: BLE001
        log(f"prueba de prioridad al carrito omitida: {str(e)[:200]}")

    # 5) ¿el modelo nuevo es mejor que el que está en producción? Se comparan los dos sobre feedback que
    #    NINGUNO ha visto al entrenar (posterior a cuando se entrenó el actual). Si no gana, no se despliega.
    verdict = champion_check(Xv, options[combine], yv_b, imp_v, is_fb[va] & (ts_all[va] > 0), ts_all[va])
    metrics["campeon"] = verdict
    log(f"comparación con el modelo en producción: {verdict}")
    if not verdict["desplegar"]:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        (MODEL_DIR / f"metrics_{version}_rechazado.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        log(f"modelo {version} NO desplegado: el actual ordena mejor la compra. Se mantiene. "
            f"Total {time.time() - t_start:.0f}s")
        return

    # 6) reentreno con todos los datos y guardado atómico (el recomendador lo recarga en caliente)
    log("reentrenando con todos los datos…")
    f_click = fit(X, yc, w, None, None, None, p_click, fixed_rounds=int(it_click * 1.1) + 1)
    bundle["click"] = f_click.model_to_string()
    if m_buy is not None:
        allc = yc == 1
        f_buy = fit(X[allc], yb[allc], w[allc], None, None, None, p_buy, fixed_rounds=int(it_buy * 1.1) + 1)
        bundle["buy"] = f_buy.model_to_string()
    if m_dir is not None:
        f_dir = fit(X, yb, w, None, None, None, dict(p_buy, num_leaves=31, min_data_in_leaf=80),
                    fixed_rounds=int(it_dir * 1.1) + 1)
        bundle["direct"] = f_dir.model_to_string()
    if m_rank is not None and combine == "rank":
        f_rank = fit_rank(X[has_pos], grade[has_pos], w[has_pos], imp[has_pos], None, None, None, None, threads,
                          fixed_rounds=int(it_rank * 1.1) + 1)
        bundle["rank"] = f_rank.model_to_string()
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
