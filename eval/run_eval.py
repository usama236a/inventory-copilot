"""Score the agent on eval/questions.json. Run: python -m eval.run_eval  (needs Ollama running)."""
import json
from pathlib import Path

import pandas as pd

from app import agent, tools

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))


def _rows(df):
    """Order- and column-name-insensitive view of a result set."""
    return sorted(tuple(round(v, 2) if isinstance(v, float) else v for v in r)
                  for r in df.itertuples(index=False))


def _correct(q, r):
    res = r["result"]
    if q["tool"] == "sql":  # execution accuracy: same rows as the gold query
        return _rows(pd.DataFrame(res)) == _rows(tools.run_sql(q["gold_sql"]))
    if q["tool"] == "forecast":
        arg = r.get("arg")
        prod = arg.get("product", "") if isinstance(arg, dict) else str(arg or "")
        return prod.lower() == q["product"].lower()
    return any(c["source"] == q["source"] for c in res)  # docs: right file retrieved


def evaluate() -> pd.DataFrame:
    rows = []
    for q in QUESTIONS:
        r = agent.ask(q["q"])
        try:
            ok = r["tool"] == q["tool"] and _correct(q, r)
        except Exception:
            ok = False
        rows.append({"question": q["q"], "tool": q["tool"], "routed": r["tool"] == q["tool"], "correct": ok})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = evaluate()
    summary = df.groupby("tool")[["routed", "correct"]].mean().round(2)
    summary.loc["ALL"] = df[["routed", "correct"]].mean().round(2)
    print(summary.to_markdown())
    Path(__file__).with_name("results.md").write_text(
        "| tool | routing acc | answer acc |\n|---|---|---|\n"
        + "\n".join(f"| {i} | {r.routed} | {r.correct} |" for i, r in summary.iterrows())
        + "\n\n## Failures\n" + ("\n".join(f"- {q}" for q in df[~df.correct].question) or "none") + "\n",
        encoding="utf-8")
