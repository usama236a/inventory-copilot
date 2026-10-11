"""Does the LightGBM forecaster beat a naive average? Run: python -m eval.forecast_bench (no LLM needed)."""
import pandas as pd

from app import tools

with tools._connect() as con:
    names = pd.read_sql_query("SELECT name FROM products", con).name
res = pd.DataFrame([tools.forecast(n, 14) for n in names])
print(f"products: {len(res)}")
print(f"mean MAE   LightGBM: {res.holdout_mae.mean():.2f}   naive 28-day avg: {res.naive_mae.mean():.2f}")
print(f"LightGBM wins on {(res.holdout_mae < res.naive_mae).mean():.0%} of products")
