"""Evaluate the live multi-agent system on the golden question set.

What it is: a fixed list of farmer questions (golden_set.jsonl), each with the expert(s) that
should answer it and a few words a good answer should contain. The script sends every question
through the real orchestrator and real A2A experts, then scores the results.

Why it matters: prompts, models and agents change all the time. A change that looks harmless
(a new Gemini model, a reworded prompt) can quietly send fertilizer questions to the help desk.
Running this before every release - and failing the release when accuracy drops - is the
"quality gate" that catches that. It is the agent version of a model's test-set accuracy.

Scores:
  routing_accuracy   the set of experts asked == the expected set ("blocked" for injections,
                     none for off-topic)
  keyword_hit_rate   the answer contains at least one expected keyword (a cheap relevance check)
  error_rate         turns that ended in an error / unavailable state
  latency p50 / p95  end-to-end seconds per question

Results are saved to data/evals/eval_<time>.json and, when MLflow is installed and
MLFLOW_TRACKING_URI is set, logged as an MLflow run (experiment "tea-agent-evals"), next to the
ML models' training runs. Exit code 1 when routing_accuracy < EVAL_MIN_ROUTING_ACCURACY.

Run (experts must be running):  python "agent (crew AI , langchain)/run_hybrid.py" eval
Options: --limit N, --only fer,hlp (id prefixes), --delay SECONDS (Gemini free tier: keep >= 15)
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agent.config import settings

GOLDEN_SET = Path(__file__).with_name("golden_set.jsonl")
REPORT_DIR = Path("data/evals")
EVAL_USER_ID = 3


def load_cases(limit: int = 0, only: str = "") -> List[Dict[str, Any]]:
    cases = [json.loads(line) for line in GOLDEN_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if only:
        prefixes = tuple(p.strip() for p in only.split(",") if p.strip())
        cases = [c for c in cases if c["id"].startswith(prefixes)]
    return cases[:limit] if limit else cases


def actual_route(intent: str) -> set:
    if intent in ("unknown", ""):
        return set()
    return set(intent.split("+"))


def score_case(case: Dict[str, Any], result: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    intent = result.get("intent", "")
    response = (result.get("response") or "").lower()
    keywords = [k.lower() for k in case.get("keywords", [])]
    return {
        "id": case["id"],
        "expected": sorted(case["expected"]),
        "actual": sorted(actual_route(intent)),
        "route_ok": actual_route(intent) == set(case["expected"]),
        "keyword_ok": (not keywords) or any(k in response for k in keywords),
        "error": intent in ("error", "unavailable"),
        "seconds": round(seconds, 2),
        "experts": result.get("experts", []),
        "response": result.get("response", ""),
    }


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows) or 1
    latencies = sorted(r["seconds"] for r in rows) or [0.0]
    frameworks: Dict[str, int] = {}
    for r in rows:
        for e in r["experts"]:
            fw = e.get("framework", "?")
            frameworks[fw] = frameworks.get(fw, 0) + 1
    return {
        "cases": len(rows),
        "routing_accuracy": round(sum(r["route_ok"] for r in rows) / n, 3),
        "keyword_hit_rate": round(sum(r["keyword_ok"] for r in rows) / n, 3),
        "error_rate": round(sum(r["error"] for r in rows) / n, 3),
        "latency_p50_s": round(statistics.median(latencies), 2),
        "latency_p95_s": round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 2),
        "expert_calls_by_framework": frameworks,
    }


def log_to_mlflow(summary: Dict[str, Any], report_path: Path) -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed; skipping MLflow logging.")
        return
    mlflow.set_experiment("tea-agent-evals")
    with mlflow.start_run(run_name=report_path.stem):
        mlflow.log_params({"llm_provider": settings.LLM_PROVIDER, "llm_model": settings.GEMINI_MODEL,
                           "experts": ",".join(settings.a2a_agent_urls)})
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(report_path))
    print("Logged to MLflow experiment 'tea-agent-evals'.")


async def run(cases: List[Dict[str, Any]], delay: float) -> List[Dict[str, Any]]:
    from agent.agents.orchestrator.orchestrator_agent import OrchestratorAgent

    orchestrator = OrchestratorAgent()
    rows = []
    for i, case in enumerate(cases):
        start = time.perf_counter()
        result = await orchestrator.process_chat(case["message"], EVAL_USER_ID, trace_id=f"eval-{case['id']}")
        row = score_case(case, result, time.perf_counter() - start)
        rows.append(row)
        mark = "PASS" if row["route_ok"] else "FAIL"
        print(f"[{mark}] {case['id']:<7} expected={row['expected']} actual={row['actual']} {row['seconds']}s")
        if delay and i < len(cases) - 1:
            await asyncio.sleep(delay)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--delay", type=float, default=15.0)
    args = parser.parse_args()

    rows = asyncio.run(run(load_cases(args.limit, args.only), args.delay))
    summary = summarise(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"eval_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    report_path.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {report_path}")
    log_to_mlflow(summary, report_path)

    if summary["routing_accuracy"] < settings.EVAL_MIN_ROUTING_ACCURACY:
        print(f"FAILED quality gate: routing_accuracy {summary['routing_accuracy']} "
              f"< {settings.EVAL_MIN_ROUTING_ACCURACY}")
        sys.exit(1)


if __name__ == "__main__":
    main()
