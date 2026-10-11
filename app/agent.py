"""Router agent: the LLM picks a tool, we run it, the LLM explains the result."""
import json
import re

from . import llm, tools

SCHEMA = """SQLite tables:
products(id, name, category, supplier_id, unit_cost, reorder_level)
suppliers(id, name, lead_days)
stock(product_id, qty)   -- current units on hand
sales(date, product_id, qty, revenue)   -- one row per product per day, date as 'YYYY-MM-DD', 2025-10-01 to 2026-09-30
Categories: Electronics, Stationery, Home, Grocery."""

ROUTER = f"""You are an inventory assistant. Pick ONE tool and reply with JSON only.
Tools:
- sql: stock levels, sales history, revenue, products, suppliers. arg = one read-only SQLite SELECT.
- forecast: future demand, or whether a product will run out. arg = {{"product": "<exact product name>", "days": <int>}}
- docs: policies, returns, supplier terms, reorder rules. arg = a search query string.
{SCHEMA}
Reply format: {{"tool": "sql" | "forecast" | "docs", "arg": ...}}
Question: """


def _parse_json(text: str) -> dict:
    """Robustly parse JSON from LLM output, stripping markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _run(tool, arg):
    if tool == "sql":
        if isinstance(arg, dict):
            arg = arg.get("query") or arg.get("sql") or next(iter(arg.values()), "")
        return tools.run_sql(str(arg)).to_dict("records")
    if tool == "forecast":
        if isinstance(arg, str):
            arg = {"product": arg, "days": 14}
        prod = arg.get("product") or arg.get("product_name") or arg.get("name") or str(arg)
        days = int(arg.get("days", 14))
        return tools.forecast(prod, days)
    if tool == "docs":
        if isinstance(arg, dict):
            arg = arg.get("query") or arg.get("question") or arg.get("search") or str(list(arg.values())[0])
        return tools.search_docs(str(arg))
    raise ValueError(f"unknown tool {tool!r}")


def _plan(prompt):
    raw = llm.chat(prompt, json_mode=True)
    plan = _parse_json(raw)
    return str(plan["tool"]).lower().strip(), plan.get("arg")


def ask(question: str) -> dict:
    prompt = ROUTER + question
    try:
        tool, arg = _plan(prompt)
        try:
            result = _run(tool, arg)
        except Exception as e:
            if tool != "sql":
                raise
            # one self-repair attempt: show the model its own SQL error
            _, arg = _plan(f"{prompt}\nYour previous SQL failed with: {e}\nReturn corrected JSON.")
            result = _run(tool, arg)
    except Exception as e:
        return {"tool": None, "arg": None, "result": {"error": str(e)},
                "answer": f"Sorry, I couldn't answer that ({e})."}

    try:
        answer = llm.chat(
            f"Question: {question}\nTool used: {tool}\nResult: {json.dumps(result, default=str)[:3000]}\n"
            "Answer in 1-3 plain sentences using only this result.").strip()
    except Exception as e:
        answer = f"Found data using {tool}, but could not format answer: {e}"

    return {"tool": tool, "arg": arg, "result": result, "answer": answer}

