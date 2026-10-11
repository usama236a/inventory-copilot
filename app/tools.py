"""The three tools the agent can call: SQL, demand forecast, policy search."""
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.getenv("INVENTORY_DB", ROOT / "data" / "inventory.db"))
DOCS = ROOT / "docs"
BLOCKED = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


def _connect():
    if not DB.exists():
        from data.make_data import build
        build(DB)
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)  # read-only at the database level


# ---------- 1. SQL ----------
def run_sql(sql: str, limit: int = 50) -> pd.DataFrame:
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.I)
        sql = re.sub(r"\s*```$", "", sql)
    sql = sql.strip().rstrip(";")
    if not re.match(r"(?is)^(select|with)\b", sql) or ";" in sql or BLOCKED.search(sql):
        raise ValueError("Only a single read-only SELECT is allowed")
    with _connect() as con:
        return pd.read_sql_query(sql, con).head(limit)



# ---------- 2. Forecast ----------
def _row(h, date):
    """Features for one day, given the history h of daily quantities."""
    return [h[-1], h[-7], h[-14], np.mean(h[-7:]), np.mean(h[-28:]), date.dayofweek]


def _fit_predict(y: pd.Series, days: int) -> np.ndarray:
    """Train LightGBM on lag features, then forecast `days` ahead recursively."""
    h = list(y.to_numpy(dtype=float))
    X = np.array([_row(h[:i], y.index[i]) for i in range(28, len(h))])
    model = lgb.LGBMRegressor(objective="poisson", n_estimators=150, learning_rate=0.05,
                              num_leaves=15, verbose=-1).fit(X, h[28:])
    out, date = [], y.index[-1]
    for _ in range(days):
        date += pd.Timedelta(days=1)
        h.append(float(model.predict(np.array([_row(h, date)]))[0]))
        out.append(h[-1])
    return np.array(out)


def forecast(product: str, days: int = 14) -> dict:
    product_clean = str(product).strip().strip("'\"")
    with _connect() as con:
        p = pd.read_sql_query(
            "SELECT p.id, p.name, s.qty AS stock, sp.lead_days FROM products p "
            "JOIN stock s ON s.product_id = p.id JOIN suppliers sp ON sp.id = p.supplier_id "
            "WHERE lower(p.name) = lower(?)", con, params=[product_clean])
        if p.empty:
            raise ValueError(f"Unknown product: {product}")
        p = p.iloc[0]
        s = pd.read_sql_query("SELECT date, qty FROM sales WHERE product_id = ?", con, params=[int(p.id)])
    y = s.set_index(pd.to_datetime(s.date)).qty.asfreq("D", fill_value=0)

    # honest check: forecast the last 14 known days and compare with a naive average
    actual = y.iloc[-14:].to_numpy()
    mae = float(np.abs(_fit_predict(y.iloc[:-14], 14) - actual).mean())
    naive_mae = float(np.abs(y.iloc[-42:-14].mean() - actual).mean())

    fc = _fit_predict(y, max(days, int(p.lead_days)))
    cum = np.cumsum(fc)
    stockout = int(np.argmax(cum >= p.stock)) + 1 if cum[-1] >= p.stock else None
    return {
        "product": p["name"], "stock": int(p.stock), "lead_days": int(p.lead_days),
        f"forecast_units_next_{days}d": round(float(fc[:days].sum()), 1),
        "stockout_in_days": stockout,
        "reorder_now": bool(stockout is not None and stockout <= p.lead_days),
        "holdout_mae": round(mae, 2), "naive_mae": round(naive_mae, 2),
    }


# ---------- 3. Policy search ----------
@lru_cache(maxsize=1)
def _index():
    chunks = [(f.name, part.strip()) for f in sorted(DOCS.glob("*.md"))
              for part in re.split(r"\n(?=#)", f.read_text(encoding="utf-8")) if part.strip()]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit([c[1] for c in chunks])
    return chunks, vec, vec.transform([c[1] for c in chunks])


def search_docs(query: str, k: int = 2) -> list[dict]:
    chunks, vec, matrix = _index()
    sims = cosine_similarity(vec.transform([query]), matrix)[0]
    return [{"source": chunks[i][0], "text": chunks[i][1], "score": round(float(sims[i]), 3)}
            for i in sims.argsort()[::-1][:k]]
