"""Post-deploy smoke test: is every expert up, ready, discoverable and protected?

Run it right after a deploy (CI does this after `docker compose up`). It checks, per expert:
  1. the Agent Card loads and its first skill id is set (the orchestrator needs it)
  2. /healthz answers 200 (process alive)
  3. /readyz answers 200 (LLM key, model files and databases usable)
  4. when A2A_AUTH_TOKEN is set: a call without the token is refused with 401
With --ask it also sends one real question to each expert over A2A (this costs LLM calls).

Exit code 0 = all good, 1 = something failed, so a deploy pipeline can stop or roll back.

    python "agent (crew AI , langchain)/run_hybrid.py" smoke [--ask]
"""
import argparse
import asyncio
import sys

import httpx

from agent.config import settings

SAMPLE_QUESTIONS = {
    "disease": "My tea leaves have small pale blisters. What is it?",
    "fertilize": "How much fertilizer should I use?",
    "company_help": "Which days does the company collect green leaves?",
}


async def check_expert(http: httpx.AsyncClient, url: str) -> bool:
    ok = True

    def report(passed: bool, what: str) -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'OK' if passed else 'FAIL'}] {what}")

    print(url)
    try:
        card = (await http.get(f"{url}/.well-known/agent-card.json")).json()
        report(bool(card.get("skills")), f"agent card: {card.get('name')} (skill={card['skills'][0]['id']})")
    except Exception as e:
        report(False, f"agent card: {e}")
        return False

    health = await http.get(f"{url}/healthz")
    report(health.status_code == 200, "liveness /healthz")
    ready = await http.get(f"{url}/readyz")
    failed = [k for k, v in ready.json().get("checks", {}).items() if not v["ok"]]
    report(ready.status_code == 200, f"readiness /readyz {'' if not failed else 'failing: ' + ', '.join(failed)}")

    if settings.A2A_AUTH_TOKEN:
        anon = await http.post(f"{url}/a2a", json={})  # this client sends no token
        report(anon.status_code == 401, f"auth enforced (no token -> {anon.status_code})")
    else:
        print("  [WARN] A2A_AUTH_TOKEN not set - experts accept unauthenticated calls")
    return ok


async def ask_each() -> bool:
    from agent.agents.orchestrator.a2a_client import AgentRegistry
    from agent.state.agent_state import OrchestratorContext

    registry = AgentRegistry(settings.a2a_agent_urls)
    ok = True
    for agent in await registry.agents():
        question = SAMPLE_QUESTIONS.get(agent.intent, "Hello")
        text, meta = await registry.call(agent, question, OrchestratorContext(user_id=3, trace_id="smoke"))
        passed = meta.get("status") == "ok"
        ok = ok and passed
        print(f"  [{'OK' if passed else 'FAIL'}] {agent.card.name} ({meta.get('framework')}) "
              f"{meta.get('latency_ms')} ms tools={meta.get('tools_used')}: {text[:80]!r}")
    return ok


async def main(ask: bool) -> int:
    async with httpx.AsyncClient(timeout=10) as http:
        results = [await check_expert(http, url) for url in settings.a2a_agent_urls]
    if ask:
        print("Sending one question to each expert...")
        results.append(await ask_each())
    print("SMOKE TEST", "PASSED" if all(results) else "FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", action="store_true", help="also send a real question to each expert")
    sys.exit(asyncio.run(main(parser.parse_args().ask)))
