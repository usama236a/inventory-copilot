import json

import pytest

from app import llm, tools
from eval import run_eval


@pytest.mark.parametrize("bad", [
    "DROP TABLE products", "DELETE FROM sales", "SELECT 1; DROP TABLE stock",
    "UPDATE stock SET qty = 0", "PRAGMA table_info(sales)",
])
def test_sql_guard_blocks_writes(bad):
    with pytest.raises(ValueError):
        tools.run_sql(bad)


def test_sql_select_works():
    assert tools.run_sql("SELECT COUNT(*) AS n FROM products").n[0] == 32


def test_database_is_read_only():
    with tools._connect() as con, pytest.raises(Exception):
        con.execute("DELETE FROM sales")


def test_forecast_shape():
    f = tools.forecast("Earbuds", 14)
    assert f["forecast_units_next_14d"] > 0 and f["holdout_mae"] >= 0
    assert isinstance(f["reorder_now"], bool)


def test_forecast_unknown_product():
    with pytest.raises(ValueError):
        tools.forecast("Flying Carpet")


@pytest.mark.parametrize("q", [x for x in run_eval.QUESTIONS if x["tool"] == "docs"], ids=lambda q: q["q"])
def test_docs_retrieval(q):
    assert tools.search_docs(q["q"])[0]["source"] == q["source"]


def test_eval_harness_with_oracle_llm(monkeypatch):
    """With a perfect fake LLM the pipeline itself must score 100% on SQL and routing."""
    def oracle(prompt, json_mode=False):
        if "Answer in 1-3" in prompt:
            return "ok"
        q = next(x for x in run_eval.QUESTIONS if prompt.endswith(x["q"]))
        arg = {"sql": q.get("gold_sql"), "docs": q["q"], "forecast": {"product": q.get("product"), "days": 14}}[q["tool"]]
        return json.dumps({"tool": q["tool"], "arg": arg})

    monkeypatch.setattr(llm, "chat", oracle)
    df = run_eval.evaluate()
    assert df.routed.all()
    assert df[df.tool == "sql"].correct.all()
    assert df.correct.mean() >= 0.95


def test_sql_markdown_block():
    df = tools.run_sql("```sql\nSELECT COUNT(*) AS n FROM products;\n```")
    assert df.n[0] == 32


def test_json_parsing_resilience():
    from app.agent import _parse_json
    assert _parse_json('{"tool": "sql", "arg": "SELECT 1"}') == {"tool": "sql", "arg": "SELECT 1"}
    assert _parse_json('```json\n{"tool": "sql", "arg": "SELECT 1"}\n```') == {"tool": "sql", "arg": "SELECT 1"}


def test_forecast_quoted_product():
    f = tools.forecast("'Earbuds'", 7)
    assert f["product"] == "Earbuds"

