"""features_cache.py · caché de features en RAM para el recomendador en dos fases (team_40).

Qué hace
--------
1. Lanza UNA vez las queries pesadas a RawTree (popularidad, co-views, view→purchase, CTR del
   carrusel antiguo, historial de usuarios) y construye embeddings de texto (TF-IDF + SVD) del
   catálogo. Todo queda en arrays de numpy / diccionarios: el acceso en producción es < 1 ms.
2. Mantiene el estado vivo (stock, precios, productos nuevos, tendencias, co-views en directo,
   compras recientes) con métodos refresh_* que el recomendador llama desde hilos en segundo plano.
3. Expone la MISMA lógica de candidatos y de features a entrenamiento (train_model.py) y a
   producción (recommender_v2.py): no hay "training/serving skew".

Uso
---
    python features_cache.py --check     # prueba rápida de las queries (2 s cada una)
    python features_cache.py --build     # construye y guarda cache/live_cache.pkl (1-5 min)

`FeatureCache.build(until='YYYY-MM-DD')` restringe el histórico a antes de esa fecha: es lo que usa
el entrenamiento para que las features sean "point-in-time" (sin fuga de información del futuro).
"""
from __future__ import annotations

import atexit
import calendar
import json
import math
import os
import pickle
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests

# ----------------------------------------------------------------------------------------------
# Entorno y cliente HTTP
# ----------------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent


def _load_env() -> None:
    for p in (Path(".env"), HERE / ".env", HERE.parent / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_env()
API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ.get("TEAM_ID", "team_40")
CACHE_DIR = Path(os.environ.get("CACHE_DIR", HERE / "cache"))
LIVE_CACHE_PATH = CACHE_DIR / "live_cache.pkl"

_tls = threading.local()


# --- Registro NO bloqueante -------------------------------------------------------------------
# En Windows, seleccionar texto en la consola (para copiarlo) congela cualquier print() hasta que se
# pulsa Enter: si el bucle principal imprimiera directamente, dejaría de responder peticiones durante
# minutos. Aquí los mensajes van a una cola; un hilo aparte los escribe en logs/<script>.log y en la
# consola. Si la consola se congela, solo se para ese hilo: el recomendador sigue respondiendo.
_log_q: "queue.Queue[str]" = queue.Queue(maxsize=20000)
LOG_DIR = Path(os.environ.get("LOG_DIR", HERE / "logs"))


def _log_writer() -> None:
    fh = None
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = open(LOG_DIR / (Path(sys.argv[0] or "python").stem + ".log"), "a", encoding="utf-8")
    except Exception:  # noqa: BLE001
        fh = None
    while True:
        line = _log_q.get()
        if fh is not None:
            try:
                fh.write(line + "\n")
                fh.flush()
            except Exception:  # noqa: BLE001
                pass
        try:
            print(line, flush=True)
        except Exception:  # noqa: BLE001
            pass
        _log_q.task_done()


threading.Thread(target=_log_writer, name="log", daemon=True).start()


def _log_drain() -> None:
    t = time.time()
    while _log_q.unfinished_tasks and time.time() - t < 3:
        time.sleep(0.02)


atexit.register(_log_drain)


def log(msg: str) -> None:
    try:
        _log_q.put_nowait(f"[{time.strftime('%H:%M:%S')}] {msg}")
    except queue.Full:
        pass


def _http() -> requests.Session:
    """Una sesión HTTP por hilo (requests.Session no es thread-safe)."""
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"
        _tls.session = s
    return s


def query(sql: str, timeout: float = 10.0, name: str = "") -> list[dict]:
    """Ejecuta SQL en RawTree y devuelve las filas como lista de dicts. `name` solo sirve para logs."""
    r = _http().post(f"{API}/v1/query", json={"sql": sql}, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"query {name or '?'} HTTP {r.status_code}: {r.text[:400]}")
    return r.json()["data"]


def insert(table: str, rows: list[dict], timeout: float = 5.0) -> None:
    """Inserción en lote en team_NN.<table>."""
    r = _http().post(f"{API}/v1/tables/{table}", params={"database": TEAM}, json=rows, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"insert {table} HTTP {r.status_code}: {r.text[:300]}")


def query_chunked(template: str, key: str, n: int, take: int | None = None, timeout: float = 600.0,
                  name: str = "", fallback: str | None = None, workers: int = 3) -> list[dict]:
    """Descarga un resultado grande en `n` trozos deterministas: `{chunk}` → cityHash64(key) % n = i.

    `take` descarga solo los primeros trozos (muestra aleatoria reproducible de take/n).
    Si el hash no está soportado se lanza `fallback` (o la query sin trocear).
    """
    idxs = list(range(n if take is None else min(take, n)))

    def one(i: int) -> list[dict]:
        return query(template.replace("{chunk}", f"cityHash64({key}) % {n} = {i}"), timeout, name=f"{name}[{i}/{n}]")

    try:
        out = list(one(idxs[0]))
    except Exception as e:  # noqa: BLE001
        log(f"  {name}: troceado no disponible ({str(e)[:160]}); uso la query alternativa")
        return query(fallback or template.replace("{chunk}", "1 = 1"), timeout, name=f"{name}[fallback]")
    if len(idxs) > 1:
        with ThreadPoolExecutor(workers) as ex:
            for rows in ex.map(one, idxs[1:]):
                out.extend(rows)
    return out


# ----------------------------------------------------------------------------------------------
# Utilidades de parseo (RawTree es schemaless: cada valor puede llegar con tipos distintos)
# ----------------------------------------------------------------------------------------------
def parse_ts(x: Any) -> float | None:
    """ISO ('2026-11-14T16:02:11.482Z', '2026-11-14 16:02:11', '2026-11-14') o epoch → epoch UTC."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v / 1000.0 if v > 1e11 else v
    s = str(x).strip()
    if len(s) < 10:
        return None
    try:
        y, mo, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
        h = mi = 0
        sec = 0.0
        if len(s) >= 16:
            h, mi = int(s[11:13]), int(s[14:16])
        if len(s) >= 19:
            tail = s[17:].rstrip("Z")
            for sep in ("+", "-"):
                if sep in tail:
                    tail = tail.split(sep)[0]
            sec = float(tail) if tail else 0.0
        return calendar.timegm((y, mo, d, h, mi, 0)) + sec
    except (ValueError, IndexError):
        return None


def fmt_ts(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(epoch))


def as_bool(x: Any, default: bool = True) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return x != 0
    return str(x).strip().lower() in ("1", "true", "t", "yes")


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def as_obj(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.startswith("{"):
        try:
            return json.loads(x)
        except ValueError:
            return {}
    return {}


def as_list(x: Any) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str) and x.startswith("["):
        try:
            return json.loads(x)
        except ValueError:
            return []
    return []


def as_pid(x: Any) -> str:
    """'' / None / NaN → '' (sin clic / sin compra)."""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    s = str(x)
    return "" if s in ("nan", "None", "null") else s


# ----------------------------------------------------------------------------------------------
# Definición de features (compartida por entrenamiento y producción)
# ----------------------------------------------------------------------------------------------
ITEM_FEATS = ["price_log", "price_rel_cat", "price_rel_sub", "pop_view", "pop_purch", "conv_rate",
              "cart_rate", "car_ctr", "car_cvr", "self_buy", "cat_code", "is_new_item"]
DYN_FEATS = ["recent_pop", "trend_lift", "item_age_days"]
INTER_FEATS = ["in_views", "view_recency", "is_last_view", "cov_score", "buy_score", "text_max",
               "text_last", "attr_max", "attr_last", "same_cat_last", "same_sub_last", "same_brand_last", "frac_cat", "frac_sub",
               "price_vs_views", "user_cat_aff", "user_bought",
               # perfil del usuario: lo que ha comprado (histórico + en vivo) y sus carritos recientes
               "u_n_buys", "u_emb_sim", "u_emb_max", "u_attr_sim", "u_sub_frac", "u_brand_frac", "u_price_rel",
               "u_cov_score", "lc_n", "lc_in", "lc_recency", "lc_emb_max", "lc_sub_frac",
               # filtrado colaborativo a nivel de USUARIO (quién compró qué en 60 días), no de sesión
               "ucf_buy", "scf_buy", "mf_user", "mf_sess", "user_sub_aff", "user_brand_aff"]
REQ_FEATS = ["device", "region", "page", "hour", "dow", "n_views", "secs_since_last_view", "user_known", "slot"]
FEATURES = ITEM_FEATS + DYN_FEATS + INTER_FEATS + REQ_FEATS
CATEGORICAL = ["cat_code", "device", "region", "page"]
F = {name: i for i, name in enumerate(FEATURES)}
NI = len(ITEM_FEATS)
NF = len(FEATURES)

# Hiperparámetros del retrieval
EXTRA_CAPACITY = int(os.environ.get("EXTRA_CAPACITY", 4000))  # hueco para productos nuevos en directo
MAX_VIEWS = 12        # últimas vistas de la sesión que se usan
DECAY = 0.7           # peso de la k-ésima vista más reciente = DECAY**k
LIVE_COV_W = 1.5      # peso de los co-views en directo frente a los históricos
CART_WINDOW = 48 * 3600.0   # un carrito cuenta como intención durante 48 h
CART_HALFLIFE = 6 * 3600.0
EV_LAG = 5.0
MF_DIM = 48
CF_KEEP = 100         # vecinos de co-compra guardados por producto
MAX_PROFILE = 40      # compras del usuario usadas para similitud (las señales de conteo usan todas)
EMB_DIM = 96
K_COV, K_BUY, K_TXT, K_TREND, K_CAT, K_SUB, K_USER, K_NEW, K_GLOBAL = 20, 20, 12, 10, 8, 8, 5, 5, 20

# ----------------------------------------------------------------------------------------------
# SQL (todas las columnas son Dynamic: casts en claves de JOIN / GROUP BY)
# ----------------------------------------------------------------------------------------------
SQL_CATALOG = "SELECT * FROM shop.catalog"

SQL_PRODUCT_STATS = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'view')        AS views,
       countIf(event_type = 'add_to_cart') AS carts,
       countIf(event_type = 'purchase')    AS purchases
FROM shop.history_events{where}
GROUP BY product_id"""

SQL_COVIEWS = """
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'view'{and_until}) AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
            FROM shop.history_events WHERE event_type = 'view'{and_until}) AS b
  ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY p, q ORDER BY n DESC LIMIT 30 BY p"""

SQL_VIEW2BUY = """
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.history_events WHERE event_type = 'view'{and_until}) AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
            FROM shop.history_events WHERE event_type = 'purchase'{and_until}) AS b
  ON a.session_id = b.session_id
GROUP BY p, q ORDER BY n DESC LIMIT 25 BY p"""

SQL_SESSION_SAMPLE = """
SELECT shown, clicked, purchased
FROM shop.history_sessions
WHERE {chunk}{and_until}"""
SQL_SESSION_SAMPLE_FALLBACK = """
SELECT shown, clicked, purchased
FROM shop.history_sessions{where}
ORDER BY rand() LIMIT 150000"""

SQL_DAILY_VIEWS = """
SELECT product_id::String AS product_id, toDate(ts::DateTime64(3)) AS day, count() AS views
FROM shop.history_events
WHERE event_type = 'view' AND {chunk}
GROUP BY product_id, day"""

SQL_USER_PURCHASES = """
SELECT user_id::String AS user_id, product_id::String AS product_id
FROM shop.history_events
WHERE event_type = 'purchase' AND {chunk}{and_until}
GROUP BY user_id, product_id"""

SQL_USER_AFFINITY = """
SELECT e.user_id::String AS user_id, c.category::String AS category,
       countIf(e.event_type = 'view') + 3 * countIf(e.event_type = 'add_to_cart')
         + 5 * countIf(e.event_type = 'purchase') AS w
FROM shop.history_events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
WHERE {chunk}{and_until_e}
GROUP BY user_id, category
ORDER BY w DESC LIMIT 3 BY user_id"""

SQL_USER_SUB_AFF = """
SELECT e.user_id::String AS user_id, c.subcategory::String AS k,
       countIf(e.event_type = 'view') + 3 * countIf(e.event_type = 'add_to_cart')
         + 5 * countIf(e.event_type = 'purchase') AS w
FROM shop.history_events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
WHERE {chunk}{and_until_e}
GROUP BY user_id, k
ORDER BY w DESC LIMIT 8 BY user_id"""

SQL_USER_BRAND_AFF = """
SELECT e.user_id::String AS user_id, c.brand::String AS k,
       countIf(e.event_type = 'view') + 3 * countIf(e.event_type = 'add_to_cart')
         + 5 * countIf(e.event_type = 'purchase') AS w
FROM shop.history_events AS e
INNER JOIN shop.catalog AS c ON c.product_id::String = e.product_id::String
WHERE {chunk}{and_until_e}
GROUP BY user_id, k
ORDER BY w DESC LIMIT 5 BY user_id"""

# --- en directo ---
SQL_OUT_OF_STOCK = """
SELECT product_id::String AS product_id
FROM shop.catalog_updates
WHERE change::String IN ('out_of_stock', 'back_in_stock')
GROUP BY product_id
HAVING argMax(change::String, ts::DateTime64(3)) = 'out_of_stock'"""

SQL_UPDATES_ALL = "SELECT * FROM shop.catalog_updates"
SQL_UPDATES_SINCE = "SELECT * FROM shop.catalog_updates WHERE ts > '{since}'"

SQL_TRENDING = """
SELECT product_id::String AS product_id,
       countIf(event_type = 'view') AS v6,
       countIf(event_type = 'view' AND ts > now() - INTERVAL 1 HOUR) AS v2,
       countIf(event_type = 'view' AND ts > now() - INTERVAL 20 MINUTE) AS v20,
       countIf(event_type = 'purchase' AND ts > now() - INTERVAL 2 HOUR) AS b2
FROM shop.events
WHERE ts > now() - INTERVAL 6 HOUR
GROUP BY product_id"""
SQL_TRENDING_SIMPLE = """
SELECT product_id::String AS product_id, count() AS v2
FROM shop.events
WHERE event_type = 'view' AND ts > now() - INTERVAL 1 HOUR
GROUP BY product_id"""

SQL_LIVE_COVIEWS = """
SELECT a.product_id AS p, b.product_id AS q, count() AS n
FROM (SELECT session_id::String AS session_id, product_id::String AS product_id
      FROM shop.events WHERE event_type = 'view' AND ts > now() - INTERVAL 6 HOUR) AS a
INNER JOIN (SELECT session_id::String AS session_id, product_id::String AS product_id
            FROM shop.events WHERE event_type = 'view' AND ts > now() - INTERVAL 6 HOUR) AS b
  ON a.session_id = b.session_id
WHERE a.product_id != b.product_id
GROUP BY p, q HAVING n >= 2 ORDER BY n DESC LIMIT 20 BY p"""

SQL_USER_EVENTS = """
SELECT ts, event_type, user_id, product_id
FROM {table}
WHERE (event_type = 'add_to_cart' OR event_type = 'purchase'){and_since} AND {chunk}"""

SQL_OWN_PURCHASES = f"""
SELECT r.user_id::String AS user_id, f.purchased::String AS product_id
FROM {TEAM}.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.purchased != '' AND f.request_id != '__init__'"""

SQL_OWN_PURCHASES_TS = f"""
SELECT r.user_id::String AS user_id, f.purchased::String AS product_id, f.ts AS ts
FROM {TEAM}.feedback AS f
INNER JOIN shop.requests AS r ON r.request_id::String = f.request_id::String
WHERE f.purchased != '' AND f.request_id != '__init__'"""

SQL_FEEDBACK_BOOT = f"""
SELECT shown, clicked, purchased
FROM {TEAM}.feedback
WHERE request_id != '__init__' AND ts > now() - INTERVAL 12 HOUR AND ts < now() - INTERVAL 3 MINUTE"""

SQL_LIVE_PURCHASES_ALL = """
SELECT user_id::String AS user_id, product_id::String AS product_id
FROM shop.events WHERE event_type = 'purchase'
GROUP BY user_id, product_id"""
SQL_LIVE_PURCHASES_RECENT = """
SELECT user_id::String AS user_id, product_id::String AS product_id
FROM shop.events WHERE event_type = 'purchase' AND ts > now() - INTERVAL 5 MINUTE
GROUP BY user_id, product_id"""


def _topk(arr: np.ndarray, k: int) -> np.ndarray:
    """Índices de los k mayores valores estrictamente positivos, de mayor a menor."""
    if k <= 0 or arr.size == 0:
        return np.empty(0, np.int64)
    k = min(k, arr.size)
    idx = np.argpartition(-arr, k - 1)[:k]
    idx = idx[arr[idx] > 0]
    return idx[np.argsort(-arr[idx], kind="stable")]


class Request:
    """Petición ya parseada (índices internos de producto en vez de IDs)."""
    __slots__ = ("request_id", "ts", "user_id", "is_new_user", "device", "region", "page", "views",
                 "last_view_ts", "cold")


class Session:
    """Estado de la sesión derivado de recent_views: se calcula una vez por petición."""
    __slots__ = ("uniq", "rec_w", "last", "acc_cov", "acc_buy", "Ev", "Av", "decays", "cat_frac", "sub_frac",
                 "mean_price", "bought", "cart_idx", "cart_w", "acc_ucov", "acc_ucf", "acc_scf", "mf_u", "mf_s")


class FeatureCache:
    # ------------------------------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self.until: str | None = None
        self.built_at = time.time()
        self.status: dict[str, str] = {}
        self._lock = threading.Lock()
        self._upd_ts = 0.0
        self._purchases_loaded = False
        self.live_views6: np.ndarray | None = None

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        d.pop("_lock", None)
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)
        self._lock = threading.Lock()
        self._init_live_fb()

    def _init_live_fb(self) -> None:
        """Contadores del feedback en vivo (por producto y por slot) para corregir el CTR de los productos
        nuevos con lo que pasa de verdad en el carrusel. Se crean vacíos si la caché es de una versión anterior."""
        cap = getattr(self, "cap", 0)
        if not cap:
            return
        if getattr(self, "prior_ctr", None) is None or len(self.prior_ctr) != cap:
            self.prior_ctr = np.full(cap, np.nan, np.float32)
            self.prior_cvr = np.full(cap, np.nan, np.float32)
        self.fb_slot_imps = np.zeros((cap, 10), np.float32)
        self.fb_clicks = np.zeros(cap, np.float32)
        self.fb_buys = np.zeros(cap, np.float32)
        self.g_slot_imps = np.zeros(10, np.float64)
        self.g_slot_clicks = np.zeros(10, np.float64)
        self.g_slot_buys = np.zeros(10, np.float64)

    def save(self, path: Path | str = LIVE_CACHE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @staticmethod
    def load(path: Path | str = LIVE_CACHE_PATH) -> "FeatureCache":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if getattr(obj, "feature_names", None) != FEATURES:
            raise RuntimeError("la caché guardada usa otra definición de features: hay que reconstruirla")
        return obj

    @classmethod
    def build(cls, until: str | None = None, with_daily: bool = False, with_users: bool = True) -> "FeatureCache":
        """Queries pesadas → arrays en RAM. Cada bloque es independiente: si uno falla, el resto sigue."""
        self = cls()
        self.until = until
        self.feature_names = list(FEATURES)
        where = f" WHERE ts < '{until}'" if until else ""
        and_until = f" AND ts < '{until}'" if until else ""
        t0 = time.time()

        # 1) catálogo (imprescindible: si falla, se propaga el error)
        rows = query(SQL_CATALOG, timeout=120, name="catalog")
        self._init_items(rows)
        log(f"  catálogo: {self.n} productos, {len(self.cat_names)} categorías, {len(self.sub2code)} subcategorías")

        # 2) embeddings de texto
        self._step("embeddings de texto (TF-IDF + SVD)", lambda: self._fit_text(rows))
        self._step("atributos de producto", lambda: self._fit_attrs(rows))

        # 3) popularidad histórica
        self.views = np.zeros(self.cap, np.float32)
        self.carts = np.zeros(self.cap, np.float32)
        self.purch = np.zeros(self.cap, np.float32)
        self.best: list[str] = []
        self._step("popularidad (history_events)", lambda: self._load_product_stats(where))

        # 4) CTR / CVR por producto en el carrusel antiguo
        self.imps = np.zeros(self.cap, np.float32)
        self.clicks = np.zeros(self.cap, np.float32)
        self.buys = np.zeros(self.cap, np.float32)
        self._step("CTR/CVR del carrusel (history_sessions)", lambda: self._load_session_stats(where, and_until))

        # 5) co-views y view→purchase
        self.cov: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.buy: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cov_live: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.self_buy = np.zeros(self.cap, np.float32)
        self._step("co-views", lambda: self._load_coviews(and_until))
        self._step("view→purchase", lambda: self._load_view2buy(and_until))

        # 6) historial de usuarios
        self.user_buys: dict[str, np.ndarray] = {}
        self.user_aff: dict[str, tuple] = {}
        self.live_buys: dict[str, set[int]] = {}
        self.user_sub: dict[str, tuple] = {}
        self.user_brand: dict[str, tuple] = {}
        self.cbuy: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.MF = np.zeros((self.cap, 1), np.float32)
        self.mf_ok = False
        self.user_ev: dict[str, list[tuple[float, int, int]]] = {}   # usuario → [(ts, 0=carrito | 1=compra, idx)]
        self._ev_ts = 0.0
        if with_users:
            self._step("compras por usuario", lambda: self._load_user_purchases(and_until))
            if os.environ.get("CACHE_USER_AFF", "1") != "0":
                self._step("afinidad usuario×categoría", lambda: self._load_user_affinity(until))
                self._step("afinidad usuario×subcategoría", lambda: self._load_user_fine_aff(until, "sub"))
                self._step("afinidad usuario×marca", lambda: self._load_user_fine_aff(until, "brand"))
        self._step("CF de co-compra por usuario + factores", self._build_cf)

        # 7) vistas diarias (solo entrenamiento: features de tendencia point-in-time)
        self.daily_views: dict[str, np.ndarray] = {}
        if with_daily:
            self._step("vistas diarias", self._load_daily_views)

        # 8) features estáticas por producto + listas top
        self.views_long = self.views.copy()
        self.has_hist = self.views > 0
        self._compute_item_feats()
        self.set_recent_counts(None)
        self._rebuild_top_lists()
        self.built_at = time.time()
        log(f"  caché lista en {time.time() - t0:.0f}s · " + " · ".join(f"{k}: {v}" for k, v in self.status.items()))
        return self

    def _step(self, label: str, fn) -> None:
        t = time.time()
        try:
            info = fn()
            self.status[label] = "OK"
            log(f"  [OK]   {label} ({time.time() - t:.1f}s){' · ' + str(info) if info else ''}")
        except Exception as e:  # noqa: BLE001
            self.status[label] = "FALLO"
            log(f"  [FALLO] {label}: {str(e)[:300]}")

    # --- catálogo -----------------------------------------------------------------------------
    def _init_items(self, rows: list[dict]) -> None:
        rows = [r for r in rows if r.get("product_id") is not None]
        rows.sort(key=lambda r: str(r["product_id"]))
        self.cap = len(rows) + EXTRA_CAPACITY
        cap = self.cap
        self.n = 0
        self.pids: list[str] = []
        self.pid2idx: dict[str, int] = {}
        self.price = np.ones(cap, np.float32)
        self.base_price = np.ones(cap, np.float32)
        self.cat = np.full(cap, -1, np.int32)
        self.sub = np.full(cap, -1, np.int32)
        self.brand = np.full(cap, -1, np.int32)
        self.created = np.zeros(cap, np.float64)
        self.active = np.zeros(cap, bool)
        self.oos_cat = np.zeros(cap, bool)
        self.oos_fast = np.zeros(cap, bool)
        self.avail = np.zeros(cap, bool)
        self.item_feat = np.zeros((cap, NI), np.float32)
        self.E = np.zeros((cap, 1), np.float32)
        self.text_ok = False
        self.A = np.zeros((cap, 1), np.float32)
        self.attr_ok = False
        self.new_items: set[int] = set()
        self.new_by_sub: dict[int, list[int]] = {}
        self.cat_names = sorted({str(r.get("category")) for r in rows})
        self.cat2code = {c: i for i, c in enumerate(self.cat_names)}
        self.sub2code: dict[str, int] = {}
        self.brand2code: dict[str, int] = {}
        for r in rows:
            idx = self.n
            self._set_item_fields(idx, r)
            self.base_price[idx] = self.price[idx]
            self.pids.append(str(r["product_id"]))
            self.pid2idx[str(r["product_id"])] = idx
            self.n += 1
        n = self.n
        ok = np.isfinite(self.price[:n]) & (self.price[:n] > 0)
        gmean = float(self.price[:n][ok].mean()) if ok.any() else 1.0
        self.price[:n][~ok] = gmean
        self.global_price = gmean
        self.cat_price = np.full(len(self.cat2code) + 64, gmean, np.float32)
        self.sub_price = np.full(len(self.sub2code) + 512, gmean, np.float32)
        for code in range(len(self.cat2code)):
            m = self.cat[:n] == code
            if m.any():
                self.cat_price[code] = self.price[:n][m].mean()
        for code in range(len(self.sub2code)):
            m = self.sub[:n] == code
            if m.any():
                self.sub_price[code] = self.price[:n][m].mean()
        self._refresh_avail()
        self._init_live_fb()

    def _code(self, table: dict[str, int], value: Any, limit: int) -> int:
        key = str(value)
        code = table.get(key)
        if code is None:
            if len(table) >= limit:
                return -1
            code = len(table)
            table[key] = code
        return code

    def _set_item_fields(self, idx: int, r: dict) -> None:
        price = as_float(r.get("price_eur"))
        if math.isfinite(price) and price > 0:
            self.price[idx] = price
        cat = r.get("category")
        if cat is not None:
            self.cat[idx] = self._code(self.cat2code, cat, len(self.cat_names) + 60)
        if r.get("subcategory") is not None:
            self.sub[idx] = self._code(self.sub2code, r["subcategory"], 10 ** 6 if not hasattr(self, "sub_price") else len(self.sub_price) - 1)
        if r.get("brand") is not None:
            self.brand[idx] = self._code(self.brand2code, r["brand"], 10 ** 6)
        created = parse_ts(r.get("created_at"))
        if created:
            self.created[idx] = created
        self.active[idx] = as_bool(r.get("is_active"), True)

    @staticmethod
    def _text_of(r: dict) -> str:
        return " ".join(str(r.get(k) or "") for k in ("description", "subcategory", "subcategory", "category", "brand"))

    def _fit_text(self, rows: list[dict]) -> str:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        by_pid = {str(r["product_id"]): r for r in rows if r.get("product_id") is not None}
        texts = [self._text_of(by_pid[p]) for p in self.pids]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.6,
                                          sublinear_tf=True, max_features=50000, dtype=np.float32)
        X = self.vectorizer.fit_transform(texts)
        dim = int(max(2, min(EMB_DIM, X.shape[1] - 1, X.shape[0] - 1)))
        self.svd = TruncatedSVD(n_components=dim, random_state=0)
        Z = self.svd.fit_transform(X).astype(np.float32)
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        E = np.zeros((self.cap, dim), np.float32)
        E[: self.n] = Z
        self.E = E
        self.text_ok = True
        return f"vocabulario {X.shape[1]}, dim {dim}"

    @staticmethod
    def _attr_text(r: dict) -> str:
        """La descripción sin las palabras de marca / categoría / subcategoría: quedan los ATRIBUTOS del
        producto ("organic ingredients", "vintage look"…), que en un catálogo sintético son la huella
        de los rasgos latentes con los que el simulador decide."""
        import re

        drop = set()
        for k in ("brand", "subcategory", "category"):
            drop.update(re.findall(r"[a-z0-9]+", str(r.get(k) or "").lower()))
        words = re.findall(r"[a-z0-9]+", str(r.get("description") or "").lower())
        return " ".join(w for w in words if w not in drop)

    def _fit_attrs(self, rows: list[dict]) -> str:
        from sklearn.feature_extraction.text import TfidfVectorizer

        by_pid = {str(r["product_id"]): r for r in rows if r.get("product_id") is not None}
        texts = [self._attr_text(by_pid[p]) for p in self.pids]
        self.attr_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.35, stop_words="english",
                                               binary=True, max_features=600, dtype=np.float32)
        M = self.attr_vectorizer.fit_transform(texts).toarray().astype(np.float32)  # filas ya normalizadas (L2)
        A = np.zeros((self.cap, M.shape[1]), np.float32)
        A[: self.n] = M
        self.A = A
        self.attr_ok = True
        return f"{M.shape[1]} atributos, {float((M > 0).sum(1).mean()):.1f} por producto"

    def _embed(self, r: dict) -> np.ndarray:
        z = self.svd.transform(self.vectorizer.transform([self._text_of(r)])).astype(np.float32)[0]
        return z / max(float(np.linalg.norm(z)), 1e-9)

    # --- histórico ----------------------------------------------------------------------------
    def _load_product_stats(self, where: str) -> str:
        rows = query(SQL_PRODUCT_STATS.replace("{where}", where), timeout=600, name="product_stats")
        sales = []
        for r in rows:
            idx = self.pid2idx.get(str(r["product_id"]))
            n_buy = int(as_float(r.get("purchases"), 0))
            if n_buy > 0:
                sales.append((n_buy, str(r["product_id"])))
            if idx is None:
                continue
            self.views[idx] = as_float(r.get("views"), 0)
            self.carts[idx] = as_float(r.get("carts"), 0)
            self.purch[idx] = n_buy
        # mismo orden que el baseline: más vendidos primero, empates por product_id mayor
        self.best = [p for _, p in sorted(sales, reverse=True)]
        return f"{len(rows)} productos con actividad, {int(self.purch.sum())} compras"

    def _load_session_stats(self, where: str, and_until: str) -> str:
        rows = query_chunked(SQL_SESSION_SAMPLE.replace("{and_until}", and_until), "request_id::String", n=16,
                             take=2, name="session_sample",
                             fallback=SQL_SESSION_SAMPLE_FALLBACK.replace("{where}", where))
        get = self.pid2idx.get
        for r in rows:
            for p in as_list(r.get("shown")):
                idx = get(str(p))
                if idx is not None:
                    self.imps[idx] += 1
            c = get(as_pid(r.get("clicked")))
            if c is not None:
                self.clicks[c] += 1
            b = get(as_pid(r.get("purchased")))
            if b is not None:
                self.buys[b] += 1
        return f"{len(rows)} impresiones de muestra"

    def _pairs_to_lists(self, rows: list[dict], denom_q: np.ndarray, keep: int) -> tuple[dict, dict]:
        """Filas (p, q, n) → {p: (idx_q[], peso[])} con peso tipo coseno n / sqrt(views_p · denom_q)."""
        tmp: dict[int, dict[int, float]] = {}
        selfn: dict[int, float] = {}
        get = self.pid2idx.get
        for r in rows:
            p, q = get(str(r["p"])), get(str(r["q"]))
            if p is None or q is None:
                continue
            n = as_float(r.get("n"), 0)
            if p == q:
                selfn[p] = n
            w = n / math.sqrt((float(self.views[p]) + 5.0) * (float(denom_q[q]) + 5.0))
            tmp.setdefault(p, {})[q] = w
        out = {}
        for p, d in tmp.items():
            items = sorted(d.items(), key=lambda kv: -kv[1])[:keep]
            out[p] = (np.fromiter((q for q, _ in items), np.int32, len(items)),
                      np.fromiter((w for _, w in items), np.float32, len(items)))
        return out, selfn

    def _load_coviews(self, and_until: str) -> str:
        rows = query(SQL_COVIEWS.replace("{and_until}", and_until), timeout=900, name="coviews")
        self.cov, _ = self._pairs_to_lists(rows, self.views, keep=30)
        return f"{len(rows)} pares, {len(self.cov)} productos"

    def _load_view2buy(self, and_until: str) -> str:
        rows = query(SQL_VIEW2BUY.replace("{and_until}", and_until), timeout=900, name="view2buy")
        self.buy, selfn = self._pairs_to_lists(rows, self.purch, keep=25)
        for p, n in selfn.items():
            self.self_buy[p] = n
        return f"{len(rows)} pares, {len(self.buy)} productos"

    def _load_user_purchases(self, and_until: str) -> str:
        rows = query_chunked(SQL_USER_PURCHASES.replace("{and_until}", and_until), "user_id::String", n=4,
                             name="user_purchases")
        tmp: dict[str, list[int]] = {}
        get = self.pid2idx.get
        for r in rows:
            idx = get(str(r["product_id"]))
            if idx is not None:
                tmp.setdefault(str(r["user_id"]), []).append(idx)
        self.user_buys = {u: np.asarray(v, np.int32) for u, v in tmp.items()}
        return f"{len(rows)} pares, {len(self.user_buys)} compradores"

    def _load_user_affinity(self, until: str | None) -> str:
        sql = SQL_USER_AFFINITY.replace("{and_until_e}", f" AND e.ts < '{until}'" if until else "")
        rows = query_chunked(sql, "e.user_id::String", n=8, name="user_affinity")
        tmp: dict[str, list[tuple[int, float]]] = {}
        for r in rows:
            code = self.cat2code.get(str(r.get("category")))
            if code is not None:
                tmp.setdefault(str(r["user_id"]), []).append((code, as_float(r.get("w"), 0)))
        aff = {}
        for u, lst in tmp.items():
            tot = sum(w for _, w in lst) or 1.0
            aff[u] = tuple((c, w / tot) for c, w in sorted(lst, key=lambda cw: -cw[1])[:3])
        self.user_aff = aff
        return f"{len(aff)} usuarios"

    def _load_user_fine_aff(self, until: str | None, kind: str) -> str:
        """Gusto fino del usuario: sus subcategorías (8) o marcas (5) con más actividad en el histórico."""
        tpl = SQL_USER_SUB_AFF if kind == "sub" else SQL_USER_BRAND_AFF
        table = self.sub2code if kind == "sub" else self.brand2code
        sql = tpl.replace("{and_until_e}", f" AND e.ts < '{until}'" if until else "")
        rows = query_chunked(sql, "e.user_id::String", n=8, name=f"user_{kind}_aff")
        tmp: dict[str, list[tuple[int, float]]] = {}
        for r in rows:
            code = table.get(str(r.get("k")))
            if code is not None:
                tmp.setdefault(str(r["user_id"]), []).append((code, as_float(r.get("w"), 0)))
        out = {}
        for u, lst in tmp.items():
            tot = sum(w for _, w in lst) or 1.0
            out[u] = tuple((c, w / tot) for c, w in lst)
        if kind == "sub":
            self.user_sub = out
        else:
            self.user_brand = out
        return f"{len(out)} usuarios"

    def _build_cf(self) -> str:
        """Filtrado colaborativo clásico sobre la matriz usuario×producto de COMPRAS (60 días):
        · cbuy: "quien compró A también compró B" (coseno con shrinkage), top-100 vecinos por producto.
        · MF: factores latentes de producto (SVD truncada); el usuario / la sesión se proyectan como la
          media de los factores de lo que ha comprado / visto, así que sirve también para usuarios nuevos.
        Se calcula en local a partir de las compras por usuario ya descargadas: no cuesta ninguna query."""
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD

        if not self.user_buys:
            raise RuntimeError("sin compras por usuario")
        n = self.n
        rows_i, cols_i = [], []
        for k, arr in enumerate(self.user_buys.values()):
            rows_i.append(np.full(len(arr), k, np.int32))
            cols_i.append(arr.astype(np.int32))
        ri, ci = np.concatenate(rows_i), np.concatenate(cols_i)
        B = sparse.csr_matrix((np.ones(len(ri), np.float32), (ri, ci)), shape=(len(self.user_buys), n))
        B.data[:] = 1.0
        deg = np.asarray(B.sum(axis=0)).ravel()
        co = (B.T @ B).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        inv = 1.0 / np.sqrt(deg + 10.0)
        cbuy = {}
        indptr, indices, data = co.indptr, co.indices, co.data
        for p in range(n):
            a, b = indptr[p], indptr[p + 1]
            if a == b:
                continue
            q = indices[a:b]
            w = data[a:b] * inv[p] * inv[q]
            if len(q) > CF_KEEP:
                top = np.argpartition(-w, CF_KEEP - 1)[:CF_KEEP]
                q, w = q[top], w[top]
            cbuy[p] = (q.astype(np.int32), w.astype(np.float32))
        self.cbuy = cbuy
        dim = int(max(2, min(MF_DIM, n - 1, B.shape[0] - 1)))
        svd = TruncatedSVD(n_components=dim, random_state=0)
        svd.fit(B)
        V = (svd.components_.T * svd.singular_values_).astype(np.float32)
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        V[deg == 0] = 0.0
        MF = np.zeros((self.cap, dim), np.float32)
        MF[:n] = V
        self.MF = MF
        self.mf_ok = True
        return f"{len(cbuy)} productos con vecinos de co-compra, {co.nnz} pares, factores dim {dim}"

    def _load_daily_views(self) -> str:
        rows = query_chunked(SQL_DAILY_VIEWS, "product_id::String", n=4, name="daily_views")
        daily: dict[str, np.ndarray] = {}
        for r in rows:
            idx = self.pid2idx.get(str(r["product_id"]))
            if idx is None:
                continue
            day = str(r["day"])[:10]
            arr = daily.get(day)
            if arr is None:
                arr = daily[day] = np.zeros(self.cap, np.float32)
            arr[idx] = as_float(r.get("views"), 0)
        self.daily_views = daily
        return f"{len(daily)} días"

    # --- features estáticas por producto ---------------------------------------------------------
    def _write_price_feats(self, idx: int | slice) -> None:
        price = np.maximum(self.price[idx], 0.01)
        self.item_feat[idx, F["price_log"]] = np.log(price)
        self.item_feat[idx, F["price_rel_cat"]] = price / self.cat_price[self.cat[idx]]
        self.item_feat[idx, F["price_rel_sub"]] = price / self.sub_price[self.sub[idx]]

    def _compute_item_feats(self) -> None:
        n = self.n
        s = slice(0, n)
        tv, tp = float(self.views[s].sum()), float(self.purch[s].sum())
        ti = float(self.imps[s].sum())
        g_conv = tp / tv if tv > 0 else 0.0
        g_cart = float(self.carts[s].sum()) / tv if tv > 0 else 0.0
        g_ctr = float(self.clicks[s].sum()) / ti if ti > 0 else 0.0
        g_cvr = float(self.buys[s].sum()) / ti if ti > 0 else 0.0
        self.g_stats = {"conv": g_conv, "cart": g_cart, "ctr": g_ctr, "cvr": g_cvr}
        self._write_price_feats(s)
        ft = self.item_feat
        ft[s, F["pop_view"]] = np.log1p(self.views[s] / tv * n) if tv > 0 else 0.0
        ft[s, F["pop_purch"]] = np.log1p(self.purch[s] / tp * n) if tp > 0 else 0.0
        ft[s, F["conv_rate"]] = (self.purch[s] + 20 * g_conv) / (self.views[s] + 20)
        ft[s, F["cart_rate"]] = (self.carts[s] + 20 * g_cart) / (self.views[s] + 20)
        ft[s, F["car_ctr"]] = (self.clicks[s] + 30 * g_ctr) / (self.imps[s] + 30)
        ft[s, F["car_cvr"]] = (self.buys[s] + 30 * g_cvr) / (self.imps[s] + 30)
        ft[s, F["self_buy"]] = self.self_buy[s] / (self.views[s] + 20)
        ft[s, F["cat_code"]] = self.cat[s]
        ft[s, F["is_new_item"]] = (~self.has_hist[s]).astype(np.float32)

    def _rebuild_top_lists(self) -> None:
        n = self.n
        q = self.item_feat[:n, F["pop_purch"]] + 0.5 * self.item_feat[:n, F["pop_view"]] + 0.5 * self.recent_log[:n]
        order = np.argsort(-q, kind="stable")
        by_cat: dict[int, list[int]] = {}
        by_sub: dict[int, list[int]] = {}
        cat, sub = self.cat, self.sub
        for i in order.tolist():
            lc = by_cat.setdefault(int(cat[i]), [])
            if len(lc) < 40:
                lc.append(i)
            ls = by_sub.setdefault(int(sub[i]), [])
            if len(ls) < 25:
                ls.append(i)
        self.top_by_cat = {c: np.asarray(v, np.int64) for c, v in by_cat.items()}
        self.top_by_sub = {c: np.asarray(v, np.int64) for c, v in by_sub.items()}
        best_idx = [self.pid2idx[p] for p in self.best[:60] if p in self.pid2idx]
        cvr_top = _topk(self.item_feat[:n, F["car_cvr"]] * (self.imps[:n] >= 20), 15).tolist()
        seen: set[int] = set()
        glob = []
        for i in best_idx[:25] + cvr_top + best_idx[25:] + order[:40].tolist():
            if i not in seen:
                seen.add(i)
                glob.append(i)
        self.top_global = np.asarray(glob, np.int64)

    # ------------------------------------------------------------------------------------------
    # Estado dinámico
    # ------------------------------------------------------------------------------------------
    def _refresh_avail(self) -> None:
        with self._lock:
            self.avail = self.active & ~(self.oos_cat | self.oos_fast)

    def set_recent_counts(self, counts: np.ndarray | None, short: np.ndarray | None = None) -> None:
        """Fija la ventana 'reciente' (en vivo: vistas de las últimas 2 h; entrenando: el día anterior)."""
        n = self.n
        long_tot = float(self.views_long[:n].sum())
        tot = float(counts[:n].sum()) if counts is not None else 0.0
        recent_log = np.zeros(self.cap, np.float32)
        lift = np.ones(self.cap, np.float32)
        if tot > 0:
            eps = 0.5 / max(n, 1)
            share = counts / tot
            long_share = self.views_long / long_tot if long_tot > 0 else np.full(self.cap, 1.0 / max(n, 1), np.float32)
            recent_log = np.log1p(share * n).astype(np.float32)
            lift = np.clip((share + eps) / (long_share + eps), 0, 10).astype(np.float32)
            src = short if short is not None and float(short[:n].sum()) >= 200 else counts
            s_share = src / max(float(src[:n].sum()), 1.0)
            score = (s_share + eps) / (long_share + eps) * np.sqrt(src)
            score[n:] = 0
            score[src < 3] = 0
            self.trend_top = _topk(score[:n], 40)
        else:
            self.trend_top = np.empty(0, np.int64)
        self.recent_log, self.trend_lift = recent_log, lift

    def set_oos_fast(self, pids: Iterable[str]) -> None:
        """Roturas de stock según la query del baseline (se refresca cada segundo)."""
        arr = np.zeros(self.cap, bool)
        get = self.pid2idx.get
        for p in pids:
            idx = get(p)
            if idx is not None:
                arr[idx] = True
        self.oos_fast = arr
        self._refresh_avail()

    def is_available(self, pid: str) -> bool | None:
        idx = self.pid2idx.get(pid)
        return None if idx is None else bool(self.avail[idx])

    # --- catálogo en directo: precios, altas, stock ------------------------------------------
    def refresh_catalog(self) -> int:
        rows = None
        if self._upd_ts > 0:
            try:
                rows = query(SQL_UPDATES_SINCE.replace("{since}", fmt_ts(self._upd_ts - 20)), timeout=8,
                             name="catalog_updates_inc")
            except Exception:  # noqa: BLE001
                rows = None
        if rows is None:
            rows = query(SQL_UPDATES_ALL, timeout=30, name="catalog_updates")
        return self.apply_catalog_updates(rows)

    def apply_catalog_updates(self, rows: list[dict]) -> int:
        """Aplica en orden temporal los registros de shop.catalog_updates (cada uno trae el producto completo)."""
        parsed = []
        for r in rows:
            if r.get("product_id") is None:
                continue
            parsed.append((parse_ts(r.get("ts")) or 0.0, r))
        parsed.sort(key=lambda x: x[0])
        added = 0
        for ts, r in parsed:
            try:
                pid = str(r["product_id"])
                change = str(r.get("change") or "")
                idx = self.pid2idx.get(pid)
                if idx is None:
                    if self.n >= self.cap:
                        continue  # sin hueco: el producto simplemente no se recomienda
                    idx = self.n
                    self._set_item_fields(idx, r)
                    self.base_price[idx] = self.price[idx]
                    if not self.created[idx]:
                        self.created[idx] = ts or time.time()
                    self._impute_new_item(idx, r)
                    self.pids.append(pid)
                    self.n = idx + 1
                    self.pid2idx[pid] = idx  # se publica al final: los lectores nunca ven una fila a medias
                    self.new_items.add(idx)
                    self.new_by_sub.setdefault(int(self.sub[idx]), []).append(idx)
                    added += 1
                else:
                    self._set_item_fields(idx, r)
                    self._write_price_feats(idx)
                    if change == "new_product":
                        self.new_items.add(idx)
                if change == "out_of_stock":
                    self.oos_cat[idx] = True
                    if r.get("is_active") is None:
                        self.active[idx] = False
                elif change == "back_in_stock":
                    self.oos_cat[idx] = False
                    if r.get("is_active") is None:
                        self.active[idx] = True
                self._upd_ts = max(self._upd_ts, ts)
            except Exception as e:  # noqa: BLE001
                log(f"catalog update ignorado: {str(e)[:120]}")
        self._refresh_avail()
        return added

    def _impute_new_item(self, idx: int, r: dict) -> None:
        """Cold start de producto: hereda estadísticas y co-views de sus vecinos de texto."""
        self._write_price_feats(idx)
        self.item_feat[idx, F["cat_code"]] = self.cat[idx]
        self.item_feat[idx, F["is_new_item"]] = 1.0
        stat_cols = [F[c] for c in ("pop_view", "pop_purch", "conv_rate", "cart_rate", "car_ctr", "car_cvr", "self_buy")]
        n = self.n
        nb = np.empty(0, np.int64)
        w = np.empty(0, np.float32)
        if self.attr_ok:
            self.A[idx] = self.attr_vectorizer.transform([self._attr_text(r)]).toarray()[0]
        if self.text_ok:
            e = self._embed(r)
            self.E[idx] = e
            sims = self.E[:n] @ e
            sims[~self.has_hist[:n]] = -1
            nb = _topk(sims, 10)
            w = sims[nb]
        if nb.size == 0:  # sin texto: media de su subcategoría (o global)
            m = np.flatnonzero((self.sub[:n] == self.sub[idx]) & self.has_hist[:n])
            nb = m if m.size else np.flatnonzero(self.has_hist[:n])
            w = np.ones(nb.size, np.float32)
        if nb.size:
            w = w / w.sum()
            self.item_feat[idx, stat_cols] = w @ self.item_feat[nb][:, stat_cols]
            if getattr(self, "prior_ctr", None) is not None:
                self.prior_ctr[idx] = self.item_feat[idx, F["car_ctr"]]
                self.prior_cvr[idx] = self.item_feat[idx, F["car_cvr"]]
            self.views_long[idx] = float(w @ self.views_long[nb])
            merged: dict[int, float] = {}
            for j, wj in zip(nb.tolist(), w.tolist()):
                merged[j] = merged.get(j, 0.0) + 0.05 * wj
                lst = self.cov.get(j)
                if lst is not None:
                    for q, wq in zip(lst[0].tolist(), lst[1].tolist()):
                        merged[q] = merged.get(q, 0.0) + wj * wq
            items = sorted(merged.items(), key=lambda kv: -kv[1])[:30]
            self.cov[idx] = (np.fromiter((q for q, _ in items), np.int32, len(items)),
                             np.fromiter((x for _, x in items), np.float32, len(items)))
            if self.mf_ok:
                v = w @ self.MF[nb]
                self.MF[idx] = v / max(float(np.linalg.norm(v)), 1e-9)
            merged = {}
            for j, wj in zip(nb.tolist(), w.tolist()):
                lst = self.cbuy.get(j)
                if lst is not None:
                    for q, wq in zip(lst[0].tolist(), lst[1].tolist()):
                        merged[q] = merged.get(q, 0.0) + wj * wq
            if merged:
                items = sorted(merged.items(), key=lambda kv: -kv[1])[:CF_KEEP]
                self.cbuy[idx] = (np.fromiter((q for q, _ in items), np.int32, len(items)),
                                  np.fromiter((x for _, x in items), np.float32, len(items)))

    # --- tendencias y co-views en directo ----------------------------------------------------
    def refresh_trending(self) -> int:
        try:
            rows = query(SQL_TRENDING, timeout=20, name="trending")
        except Exception:  # noqa: BLE001
            rows = query(SQL_TRENDING_SIMPLE, timeout=20, name="trending_simple")
        v2 = np.zeros(self.cap, np.float32)
        v20 = np.zeros(self.cap, np.float32)
        v6 = np.zeros(self.cap, np.float32)
        get = self.pid2idx.get
        for r in rows:
            idx = get(str(r["product_id"]))
            if idx is None:
                continue
            v2[idx] = as_float(r.get("v2"), 0)
            v20[idx] = as_float(r.get("v20"), 0)
            v6[idx] = as_float(r.get("v6"), v2[idx])
        self.live_views6 = v6
        self.set_recent_counts(v2, short=v20)
        self._rebuild_top_lists()
        return len(rows)

    def refresh_live_coviews(self) -> int:
        rows = query(SQL_LIVE_COVIEWS, timeout=60, name="live_coviews")
        views = self.live_views6 if self.live_views6 is not None else np.zeros(self.cap, np.float32)
        tmp: dict[int, list[tuple[int, float]]] = {}
        get = self.pid2idx.get
        for r in rows:
            p, q = get(str(r["p"])), get(str(r["q"]))
            if p is None or q is None or p == q:
                continue
            n = as_float(r.get("n"), 0)
            # shrink fuerte (+20): con pocas horas de datos un par casual no debe dominar
            tmp.setdefault(p, []).append((q, n / math.sqrt((float(views[p]) + 20.0) * (float(views[q]) + 20.0))))
        live = {}
        for p, lst in tmp.items():
            lst.sort(key=lambda kv: -kv[1])
            lst = lst[:20]
            live[p] = (np.fromiter((q for q, _ in lst), np.int32, len(lst)),
                       np.fromiter((w for _, w in lst), np.float32, len(lst)))
        self.cov_live = live
        return len(live)

    def refresh_live_purchases(self) -> int:
        """Compras orgánicas en directo: quien ya compró un producto tiende a no repetirlo."""
        sql = SQL_LIVE_PURCHASES_RECENT if self._purchases_loaded else SQL_LIVE_PURCHASES_ALL
        rows = query(sql, timeout=60, name="live_purchases")
        for r in rows:
            self.add_purchase(str(r["user_id"]), str(r["product_id"]))
        self._purchases_loaded = True
        return len(rows)

    def load_user_events(self, table: str, since: str | None = None, incremental: bool = False) -> int:
        """Carritos y compras orgánicos por usuario, con su hora: es el perfil 'en vivo' del comprador.
        Entrenando se cargan de shop.history_events (features point-in-time); en producción se sigue
        shop.events cada pocos segundos, así que también hay señal para compradores nuevos."""
        sql = SQL_USER_EVENTS.replace("{table}", table)
        if incremental:
            sql = sql.replace("{and_since}", f" AND ts > '{fmt_ts(self._ev_ts - 5)}'").replace("{chunk}", "1 = 1")
            rows = query(sql, timeout=6, name="user_events_inc")
        else:
            sql = sql.replace("{and_since}", f" AND ts >= '{since}'" if since else "")
            rows = query_chunked(sql, "user_id::String", n=4, name=f"user_events:{table}")
        parsed = []
        get = self.pid2idx.get
        for r in rows:
            idx, ts = get(str(r.get("product_id"))), parse_ts(r.get("ts"))
            if idx is None or ts is None:
                continue
            parsed.append((ts, 1 if str(r.get("event_type")) == "purchase" else 0, idx, str(r.get("user_id"))))
        parsed.sort()
        ev = self.user_ev
        added = 0
        for ts, kind, idx, uid in parsed:
            lst = ev.get(uid)
            if lst is None:
                lst = ev[uid] = []
            tup = (ts, kind, idx)
            if incremental:
                if tup in lst[-12:]:
                    continue          # solape de la ventana incremental
                if len(lst) > 80:
                    del lst[:20]
            lst.append(tup)
            added += 1
        if parsed and table == "shop.events":
            self._ev_ts = max(self._ev_ts, parsed[-1][0])
        return added

    def refresh_user_events(self) -> int:
        if self._ev_ts <= 0:
            n = self.load_user_events("shop.events")
            if self._ev_ts <= 0:
                self._ev_ts = time.time() - 60
            return n
        return self.load_user_events("shop.events", incremental=True)

    def load_own_purchases(self) -> int:
        """Compras conseguidas con NUESTRAS recomendaciones (feedback ⨝ requests): también cuentan como
        'ya comprado' para ese usuario. Se cargan al arrancar; después las añade el lector de feedback."""
        rows = query(SQL_OWN_PURCHASES, timeout=60, name="own_purchases")
        for r in rows:
            self.add_purchase(str(r.get("user_id") or ""), str(r.get("product_id") or ""))
        return len(rows)

    def observe_feedback(self, shown: list, clicked: str, purchased: str) -> None:
        """Una impresión real del carrusel (nuestra o del fallback): cuenta impresiones por slot y clics/compras."""
        if getattr(self, "fb_slot_imps", None) is None:
            return
        get = self.pid2idx.get
        for s, p in enumerate(shown[:10]):
            idx = get(str(p))
            self.g_slot_imps[s] += 1
            if idx is not None:
                self.fb_slot_imps[idx, s] += 1
            if clicked and str(p) == clicked:
                self.g_slot_clicks[s] += 1
                if idx is not None:
                    self.fb_clicks[idx] += 1
            if purchased and str(p) == purchased:
                self.g_slot_buys[s] += 1
                if idx is not None:
                    self.fb_buys[idx] += 1

    def bootstrap_feedback(self) -> int:
        """Al arrancar: las últimas 12 h de feedback, para no empezar de cero los contadores."""
        rows = query(SQL_FEEDBACK_BOOT, timeout=120, name="feedback_boot")
        for r in rows:
            self.observe_feedback(as_list(r.get("shown")), as_pid(r.get("clicked")), as_pid(r.get("purchased")))
        self.refresh_new_item_ctr()
        return len(rows)

    def refresh_new_item_ctr(self) -> int:
        """CTR/CVR de los productos NUEVOS corregidos con el feedback real.
        Se compara lo observado con lo esperado según el slot donde salió cada impresión (clics sobre clics
        esperados), así que no importa si lo pusimos arriba o abajo. El resultado escala el valor heredado de
        los vecinos de texto, con shrinkage: con pocas impresiones apenas cambia."""
        if getattr(self, "fb_slot_imps", None) is None or self.g_slot_imps.sum() < 3000 or not self.new_items:
            return 0
        slot_ctr = (self.g_slot_clicks + 1.0) / (self.g_slot_imps + 50.0)
        slot_cvr = (self.g_slot_buys + 0.5) / (self.g_slot_imps + 50.0)
        idx = np.fromiter((i for i in self.new_items if i < self.n), np.int64)
        idx = idx[np.isfinite(self.prior_ctr[idx])]
        if not idx.size:
            return 0
        exp_c = self.fb_slot_imps[idx] @ slot_ctr
        exp_b = self.fb_slot_imps[idx] @ slot_cvr
        ratio_c = np.clip((self.fb_clicks[idx] + 5.0) / (exp_c + 5.0), 0.33, 3.0)
        ratio_b = np.clip((self.fb_buys[idx] + 2.0) / (exp_b + 2.0), 0.33, 3.0)
        self.item_feat[idx, F["car_ctr"]] = self.prior_ctr[idx] * ratio_c
        self.item_feat[idx, F["car_cvr"]] = self.prior_cvr[idx] * ratio_b
        return int(idx.size)

    def add_own_purchases_ts(self) -> int:
        """Entrenamiento: compras conseguidas con nuestras recomendaciones, CON su hora, como eventos de compra
        del usuario. Así 'ya comprado' en las filas de feedback coincide con lo que se aplica en producción."""
        rows = query(SQL_OWN_PURCHASES_TS, timeout=120, name="own_purchases_ts")
        get = self.pid2idx.get
        added = 0
        for r in rows:
            idx, ts, uid = get(str(r.get("product_id"))), parse_ts(r.get("ts")), str(r.get("user_id") or "")
            if idx is None or ts is None or not uid:
                continue
            self.user_ev.setdefault(uid, []).append((ts, 1, idx))
            added += 1
        return added

    def add_purchase(self, user_id: str, pid: str) -> None:
        idx = self.pid2idx.get(pid)
        if idx is not None and user_id:
            self.live_buys.setdefault(user_id, set()).add(idx)

    # ------------------------------------------------------------------------------------------
    # Serving: parseo → sesión → candidatos → features
    # ------------------------------------------------------------------------------------------
    def parse_request(self, rec: dict) -> Request:
        R = Request()
        ctx = as_obj(rec.get("context"))
        R.request_id = str(rec.get("request_id"))
        R.ts = parse_ts(rec.get("ts")) or time.time()
        R.user_id = str(rec.get("user_id") or "")
        R.is_new_user = as_bool(rec.get("is_new_user"), False)
        R.device = str(ctx.get("device", rec.get("device", "")))
        R.region = str(ctx.get("region", rec.get("region", "")))
        R.page = str(ctx.get("page", rec.get("page", "")))
        views: list[tuple[float, int, int]] = []
        get = self.pid2idx.get
        last_ts = None
        for k, v in enumerate(as_list(rec.get("recent_views"))):
            if isinstance(v, dict):
                pid, vts = v.get("product_id"), parse_ts(v.get("ts"))
            else:
                pid, vts = v, None
            idx = get(str(pid))
            if idx is None:
                continue
            views.append((vts if vts is not None else float(k), k, idx))
            if vts is not None and (last_ts is None or vts > last_ts):
                last_ts = vts
        views.sort()
        R.views = [idx for _, _, idx in views][-MAX_VIEWS:]
        R.last_view_ts = last_ts
        R.cold = R.is_new_user or any(i in self.new_items for i in R.views)
        return R

    def session(self, R: Request) -> Session:
        S = Session()
        cap = self.cap
        S.acc_cov = np.zeros(cap, np.float32)
        S.acc_buy = np.zeros(cap, np.float32)
        S.rec_w = {}
        S.cat_frac = np.zeros(len(self.cat_price) + 1, np.float32)
        S.sub_frac = np.zeros(len(self.sub_price) + 1, np.float32)
        uniq: list[int] = []
        cov, cov_live, buy = self.cov, self.cov_live, self.buy
        for k, v in enumerate(reversed(R.views)):
            d = DECAY ** k
            if v not in S.rec_w:
                S.rec_w[v] = d
                uniq.append(v)
            nb = cov.get(v)
            if nb is not None:
                S.acc_cov[nb[0]] += d * nb[1]
            nb = cov_live.get(v)
            if nb is not None:
                S.acc_cov[nb[0]] += (d * LIVE_COV_W) * nb[1]
            nb = buy.get(v)
            if nb is not None:
                S.acc_buy[nb[0]] += d * nb[1]
        S.uniq = np.asarray(uniq, np.int64)
        S.last = uniq[0] if uniq else -1
        if uniq:
            S.decays = np.asarray([S.rec_w[v] for v in uniq], np.float32)
            S.Ev = self.E[S.uniq]
            S.Av = self.A[S.uniq]
            inv = 1.0 / len(uniq)
            for v in uniq:
                S.cat_frac[self.cat[v]] += inv
                S.sub_frac[self.sub[v]] += inv
            S.cat_frac[-1] = 0.0  # código -1 (desconocido) no cuenta
            S.sub_frac[-1] = 0.0
            S.mean_price = float(self.price[S.uniq].mean())
        else:
            S.decays, S.Ev, S.Av, S.mean_price = None, None, None, 0.0
        # --- perfil del usuario en el instante de la petición (point-in-time) ---
        uid = R.user_id
        bought: set[int] = set()
        hist, live = self.user_buys.get(uid), self.live_buys.get(uid)
        if hist is not None:
            bought.update(hist.tolist())
        if live:
            bought |= live
        carts: dict[int, float] = {}
        ev = self.user_ev.get(uid)
        if ev:
            t_req, t_min = R.ts, R.ts - CART_WINDOW
            for t, kind, idx in list(ev):
                if t >= t_req - EV_LAG:   # en producción los eventos llegan con unos segundos de retraso
                    continue
                if kind == 1:
                    bought.add(idx)
                elif t >= t_min:
                    carts[idx] = t
        S.bought = np.fromiter(bought, np.int64, len(bought))
        cart_items = [(i, t) for i, t in carts.items() if i not in bought]
        S.cart_idx = np.asarray([i for i, _ in cart_items], np.int64)
        S.cart_w = np.asarray([0.5 ** ((R.ts - t) / CART_HALFLIFE) for _, t in cart_items], np.float32)
        # --- CF de co-compra a nivel de usuario y proyección en factores latentes ---
        cbuy = self.cbuy
        S.acc_scf = S.acc_ucf = S.mf_s = S.mf_u = None
        if uniq and cbuy:
            acc = np.zeros(cap, np.float32)
            for v in uniq:
                nb = cbuy.get(v)
                if nb is not None:
                    acc[nb[0]] += S.rec_w[v] * nb[1]
            S.acc_scf = acc
        if S.bought.size and cbuy:
            acc = np.zeros(cap, np.float32)
            prof = S.bought[:MAX_PROFILE]
            scale = 1.0 / np.sqrt(len(prof))
            for p in prof.tolist():
                nb = cbuy.get(p)
                if nb is not None:
                    acc[nb[0]] += scale * nb[1]
            S.acc_ucf = acc
        if self.mf_ok:
            if uniq:
                v = S.decays @ self.MF[S.uniq]
                nv = float(np.linalg.norm(v))
                if nv > 1e-6:
                    S.mf_s = v / nv
            if S.bought.size:
                v = self.MF[S.bought[:MAX_PROFILE]].sum(axis=0)
                nv = float(np.linalg.norm(v))
                if nv > 1e-6:
                    S.mf_u = v / nv
        S.acc_ucov = None
        if S.bought.size or S.cart_idx.size:
            acc = np.zeros(cap, np.float32)
            prof = S.bought[:MAX_PROFILE]
            scale = 1.0 / np.sqrt(max(len(prof), 1))
            for p in prof.tolist():
                nb = cov.get(p)
                if nb is not None:
                    acc[nb[0]] += scale * nb[1]
            for p, wt in zip(S.cart_idx.tolist(), S.cart_w.tolist()):
                for store in (cov, buy):
                    nb = store.get(p)
                    if nb is not None:
                        acc[nb[0]] += wt * nb[1]
            S.acc_ucov = acc
        return S

    def candidates(self, R: Request, S: Session, wide: bool = False) -> np.ndarray:
        """~50-90 candidatos en stock, mezclando todas las fuentes de retrieval."""
        n = self.n
        m = 3 if wide else 1        # 'wide': el triple de candidatos por fuente (~250 en total)
        parts: list[np.ndarray] = []
        if S.uniq.size:
            parts.append(S.uniq)                                   # lo que acaba de ver
            parts.append(_topk(S.acc_buy[:n], K_BUY * m))              # vieron esto → compraron aquello
            parts.append(_topk(S.acc_cov[:n], K_COV * m))              # co-vistos (histórico + en directo)
            if self.text_ok:                                       # parecidos por texto (cubre productos nuevos)
                T = (self.E[:n] @ S.Ev.T) * S.decays[None, :]
                parts.append(_topk(T.max(axis=1), K_TXT * m))
            for v in S.uniq[:2].tolist():                          # top de sus subcategorías / categorías
                tb = self.top_by_sub.get(int(self.sub[v]))
                if tb is not None:
                    parts.append(tb[:K_SUB * m])
                new = self.new_by_sub.get(int(self.sub[v]))
                if new:
                    parts.append(np.asarray(new[-K_NEW * m:], np.int64))
            tc = self.top_by_cat.get(int(self.cat[S.last]))
            if tc is not None:
                parts.append(tc[:K_CAT * m])
        if S.cart_idx.size:                                        # lo que tiene en el carrito y sus vecinos
            parts.append(S.cart_idx[-8:])
        if S.acc_ucov is not None:
            parts.append(_topk(S.acc_ucov[:n], 12 * m))
        if S.acc_ucf is not None:                                  # quien compró lo mismo que él también compró…
            parts.append(_topk(S.acc_ucf[:n], 15 * m))
        if S.acc_scf is not None:                                  # …y quien compró lo que está viendo
            parts.append(_topk(S.acc_scf[:n], 15 * m))
        if S.mf_u is not None:
            parts.append(_topk(self.MF[:n] @ S.mf_u, 10 * m))
        aff = self.user_aff.get(R.user_id)
        if aff:
            for code, _ in aff[:2]:
                tc = self.top_by_cat.get(code)
                if tc is not None:
                    parts.append(tc[:K_USER * m])
        parts.append(self.trend_top[:K_TREND * m])
        parts.append(self.top_global[:K_GLOBAL * m])
        allc = np.concatenate(parts).astype(np.int64)
        _, first = np.unique(allc, return_index=True)
        cands = allc[np.sort(first)]
        cands = cands[cands < n]
        cands = cands[self.avail[cands]]
        if S.bought.size:      # lo ya comprado casi nunca se vuelve a comprar: fuera, si queda con qué llenar
            fresh = cands[~np.isin(cands, S.bought)]
            if fresh.size >= 15:
                cands = fresh
        if cands.size < 15:  # red de seguridad: completar con lo más vendido disponible
            extra = self.top_global[self.avail[self.top_global]]
            if extra.size < 15:
                extra = np.concatenate([extra, np.flatnonzero(self.avail[:n])[:30]])
            allc = np.concatenate([cands, extra])
            _, first = np.unique(allc, return_index=True)
            cands = allc[np.sort(first)]
        return cands

    def all_candidates(self) -> np.ndarray:
        """Todo el catálogo disponible: con ~5.000 productos se puede puntuar entero (sin retrieval)."""
        return np.flatnonzero(self.avail[: self.n])

    def featurize(self, R: Request, S: Session, cands: np.ndarray, vocab: dict | None = None,
                  slots: np.ndarray | None = None) -> np.ndarray:
        """Matriz [len(cands), NF] float32 con el orden de columnas de FEATURES."""
        vocab = vocab or {}
        m = len(cands)
        X = np.zeros((m, NF), np.float32)
        X[:, :NI] = self.item_feat[cands]
        X[:, F["recent_pop"]] = self.recent_log[cands]
        X[:, F["trend_lift"]] = self.trend_lift[cands]
        created = self.created[cands]
        age = np.where(created > 0, (R.ts - created) / 86400.0, 400.0)
        X[:, F["item_age_days"]] = np.clip(age, 0, 400)
        cat_c, sub_c = self.cat[cands], self.sub[cands]
        if S.uniq.size:
            dense = np.zeros(self.cap, np.float32)
            dense[S.uniq] = S.decays
            rec = dense[cands]
            X[:, F["in_views"]] = rec > 0
            X[:, F["view_recency"]] = rec
            X[:, F["is_last_view"]] = cands == S.last
            X[:, F["cov_score"]] = S.acc_cov[cands]
            X[:, F["buy_score"]] = S.acc_buy[cands]
            if self.text_ok:
                T = self.E[cands] @ S.Ev.T
                X[:, F["text_max"]] = T.max(axis=1)
                X[:, F["text_last"]] = T[:, 0]
            if self.attr_ok:
                T = self.A[cands] @ S.Av.T
                X[:, F["attr_max"]] = T.max(axis=1)
                X[:, F["attr_last"]] = T[:, 0]
            X[:, F["same_cat_last"]] = cat_c == self.cat[S.last]
            X[:, F["same_sub_last"]] = sub_c == self.sub[S.last]
            X[:, F["same_brand_last"]] = self.brand[cands] == self.brand[S.last]
            X[:, F["frac_cat"]] = S.cat_frac[cat_c]
            X[:, F["frac_sub"]] = S.sub_frac[sub_c]
            X[:, F["price_vs_views"]] = np.log(np.maximum(self.price[cands], 0.01) / max(S.mean_price, 0.01))
        uid = R.user_id
        aff = self.user_aff.get(uid)
        if aff:
            arr = np.zeros(len(self.cat_price) + 1, np.float32)
            for code, w in aff:
                arr[code] = w
            X[:, F["user_cat_aff"]] = arr[cat_c]
        hist = self.user_buys.get(uid)
        Ec = self.E[cands] if self.text_ok and (S.bought.size or S.cart_idx.size) else None
        if S.bought.size:
            P = S.bought
            dense = np.zeros(self.cap, np.float32)
            dense[P] = 1.0
            X[:, F["user_bought"]] = dense[cands]
            X[:, F["u_n_buys"]] = np.log1p(P.size)
            prof = P[:MAX_PROFILE]
            if Ec is not None:
                Ep = self.E[prof]
                cen = Ep.mean(axis=0)
                X[:, F["u_emb_sim"]] = Ec @ (cen / max(float(np.linalg.norm(cen)), 1e-9))
                X[:, F["u_emb_max"]] = (Ec @ Ep.T).max(axis=1)
            if self.attr_ok:
                cen = self.A[prof].mean(axis=0)
                X[:, F["u_attr_sim"]] = self.A[cands] @ (cen / max(float(np.linalg.norm(cen)), 1e-9))
            cnt = np.zeros(len(self.sub_price) + 1, np.float32)
            np.add.at(cnt, self.sub[P], 1.0 / P.size)
            cnt[-1] = 0.0
            X[:, F["u_sub_frac"]] = cnt[sub_c]
            cnt = np.zeros(len(self.brand2code) + 2, np.float32)
            np.add.at(cnt, self.brand[P], 1.0 / P.size)
            cnt[-1] = 0.0
            X[:, F["u_brand_frac"]] = cnt[self.brand[cands]]
            X[:, F["u_price_rel"]] = np.log(np.maximum(self.price[cands], 0.01) / max(float(self.price[P].mean()), 0.01))
        if S.acc_ucov is not None:
            X[:, F["u_cov_score"]] = S.acc_ucov[cands]
        if S.acc_ucf is not None:
            X[:, F["ucf_buy"]] = S.acc_ucf[cands]
        if S.acc_scf is not None:
            X[:, F["scf_buy"]] = S.acc_scf[cands]
        if S.mf_u is not None or S.mf_s is not None:
            Mc = self.MF[cands]
            if S.mf_u is not None:
                X[:, F["mf_user"]] = Mc @ S.mf_u
            if S.mf_s is not None:
                X[:, F["mf_sess"]] = Mc @ S.mf_s
        fine = self.user_sub.get(uid)
        if fine:
            arr = np.zeros(len(self.sub_price) + 1, np.float32)
            for code, wt in fine:
                arr[code] = wt
            X[:, F["user_sub_aff"]] = arr[sub_c]
        fine = self.user_brand.get(uid)
        if fine:
            arr = np.zeros(len(self.brand2code) + 2, np.float32)
            for code, wt in fine:
                arr[code] = wt
            X[:, F["user_brand_aff"]] = arr[self.brand[cands]]
        if S.cart_idx.size:
            C = S.cart_idx
            X[:, F["lc_n"]] = C.size
            dense = np.zeros(self.cap, np.float32)
            dense[C] = S.cart_w
            X[:, F["lc_recency"]] = dense[cands]
            X[:, F["lc_in"]] = dense[cands] > 0
            if Ec is not None:
                X[:, F["lc_emb_max"]] = (Ec @ self.E[C[-MAX_PROFILE:]].T).max(axis=1)
            cnt = np.zeros(len(self.sub_price) + 1, np.float32)
            np.add.at(cnt, self.sub[C], 1.0 / C.size)
            cnt[-1] = 0.0
            X[:, F["lc_sub_frac"]] = cnt[sub_c]
        X[:, F["device"]] = vocab.get("device", {}).get(R.device, -1)
        X[:, F["region"]] = vocab.get("region", {}).get(R.region, -1)
        X[:, F["page"]] = vocab.get("page", {}).get(R.page, -1)
        day_secs = R.ts % 86400.0
        X[:, F["hour"]] = int(day_secs // 3600)
        X[:, F["dow"]] = (int(R.ts // 86400) + 4) % 7  # 1970-01-01 fue jueves → 0 = lunes
        X[:, F["n_views"]] = len(R.views)
        gap = -1.0 if R.last_view_ts is None else min(max(R.ts - R.last_view_ts, 0.0), 3600.0)
        X[:, F["secs_since_last_view"]] = gap
        X[:, F["user_known"]] = 0.0 if R.is_new_user else float(aff is not None or hist is not None)
        if slots is not None:
            X[:, F["slot"]] = slots
        return X


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def smoke_check() -> int:
    """Prueba cada query ligera contra RawTree e informa: útil antes de lanzar producción."""
    checks = {
        "catalog": "SELECT * FROM shop.catalog LIMIT 2",
        "requests": "SELECT request_id, ts, user_id, is_new_user, context, recent_views FROM shop.requests ORDER BY ts DESC LIMIT 2",
        "hash (troceado)": "SELECT count() AS n FROM shop.catalog WHERE cityHash64(product_id::String) % 4 = 0",
        "out_of_stock": SQL_OUT_OF_STOCK,
        "catalog_updates": SQL_UPDATES_ALL + " LIMIT 3",
        "catalog_updates incremental": SQL_UPDATES_SINCE.replace("{since}", fmt_ts(time.time() - 600)) + " LIMIT 3",
        "trending": SQL_TRENDING + " LIMIT 3",
        "trending_simple": SQL_TRENDING_SIMPLE + " LIMIT 3",
        "live_coviews": SQL_LIVE_COVIEWS.replace("LIMIT 20 BY p", "LIMIT 1 BY p LIMIT 3"),
        "live_purchases": SQL_LIVE_PURCHASES_RECENT + " LIMIT 3",
        "own_purchases": SQL_OWN_PURCHASES + " LIMIT 3",
        "own_purchases_ts": SQL_OWN_PURCHASES_TS + " LIMIT 3",
        "feedback_boot": SQL_FEEDBACK_BOOT + " LIMIT 3",
        "user_sub_aff": SQL_USER_SUB_AFF.replace("{and_until_e}", "").replace(
            "{chunk}", "cityHash64(e.user_id::String) % 3000 = 0") + " LIMIT 3",
        "user_brand_aff": SQL_USER_BRAND_AFF.replace("{and_until_e}", "").replace(
            "{chunk}", "cityHash64(e.user_id::String) % 3000 = 0") + " LIMIT 3",
        "user_events": SQL_USER_EVENTS.replace("{table}", "shop.events").replace(
            "{and_since}", f" AND ts > '{fmt_ts(time.time() - 60)}'").replace("{chunk}", "1 = 1") + " LIMIT 3",
        "user_events (histórico)": SQL_USER_EVENTS.replace("{table}", "shop.history_events").replace(
            "{and_since}", " AND ts >= '2026-11-13'").replace("{chunk}", "cityHash64(user_id::String) % 500 = 0") + " LIMIT 3",
        "feedback": f"SELECT * FROM {TEAM}.feedback WHERE request_id != '__init__' ORDER BY ts DESC LIMIT 2",
        "session_sample": SQL_SESSION_SAMPLE.replace("{chunk}", "cityHash64(request_id::String) % 100000 = 0").replace("{and_until}", ""),
    }
    bad = 0
    for name, sql in checks.items():
        t = time.time()
        try:
            rows = query(sql, timeout=120, name=name)
            sample = json.dumps(rows[0], default=str)[:230] if rows else "(sin filas)"
            log(f"[OK]    {name:28s} {len(rows):5d} filas · {time.time() - t:5.2f}s · {sample}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            log(f"[FALLO] {name:28s} {str(e)[:300]}")
    log("todo en orden" if not bad else f"{bad} queries han fallado: mira el error y ajusta el SQL correspondiente")
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(1 if smoke_check() else 0)
    log("construyendo la caché de producción (histórico completo)…")
    cache = FeatureCache.build()
    cache.save(LIVE_CACHE_PATH)
    log(f"guardada en {LIVE_CACHE_PATH} ({LIVE_CACHE_PATH.stat().st_size / 1e6:.1f} MB)")
