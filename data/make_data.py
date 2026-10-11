"""Generate a synthetic inventory database (SQLite). Run: python -m data.make_data"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path(__file__).parent / "inventory.db"
CATS = {
    "Electronics": ["USB Cable", "Charger", "Earbuds", "Power Bank", "Webcam", "Mouse", "Keyboard", "Phone Case"],
    "Stationery": ["Notebook", "Pen Pack", "Stapler", "Marker Set", "Folder", "Sticky Notes", "Highlighter", "Calculator"],
    "Home": ["Desk Lamp", "Water Bottle", "Storage Box", "Mug", "Wall Clock", "Bed Sheet", "Towel Set", "Curtain"],
    "Grocery": ["Green Tea", "Coffee Jar", "Biscuits", "Olive Oil", "Rice 5kg", "Honey", "Dates", "Almonds"],
}


def build(path=DB, seed=42):
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.unlink(missing_ok=True)

    suppliers = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alpha Traders", "Beta Wholesale", "Crescent Supply", "Delta Imports"],
        "lead_days": [3, 7, 5, 10],
    })
    products = pd.DataFrame(
        [(i + 1, n, c) for i, (c, n) in enumerate((c, n) for c, ns in CATS.items() for n in ns)],
        columns=["id", "name", "category"],
    )
    products["supplier_id"] = rng.integers(1, 5, len(products))
    products["unit_cost"] = rng.uniform(1, 40, len(products)).round(2)
    products["reorder_level"] = rng.integers(20, 80, len(products))

    days = pd.date_range("2025-10-01", "2026-09-30")
    weekly = np.array([1, 1, 1, 1, 1.2, 1.5, 1.4])[days.dayofweek.to_numpy()]  # Mon..Sun
    sales = []
    for p in products.itertuples():
        season = 1 + 0.25 * np.sin(2 * np.pi * days.dayofyear.to_numpy() / 365 + p.supplier_id)
        qty = rng.poisson(rng.uniform(2, 14) * weekly * season)
        sales.append(pd.DataFrame({
            "date": days.strftime("%Y-%m-%d"), "product_id": p.id, "qty": qty,
            "revenue": (qty * p.unit_cost * 1.4).round(2),
        }))
    sales = pd.concat(sales)
    sales = sales[sales.qty > 0]
    stock = pd.DataFrame({"product_id": products.id, "qty": rng.integers(0, 250, len(products))})

    with sqlite3.connect(path) as con:
        for name, df in [("suppliers", suppliers), ("products", products), ("stock", stock), ("sales", sales)]:
            df.to_sql(name, con, index=False)
    return path


if __name__ == "__main__":
    print(f"Created {build()}")
