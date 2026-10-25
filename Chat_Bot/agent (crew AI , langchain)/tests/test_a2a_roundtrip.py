"""End-to-end A2A tests: real servers and real client, fake experts (no LLM calls).

The A2A apps run in memory through httpx.ASGITransport, so no ports are opened. This checks the
contract between the orchestrator and every expert: discovery, metadata in both directions,
auth, photo forwarding, timeouts, validation, circuit breaker, and the health endpoints.
"""
import asyncio
from pathlib import Path

import httpx
import pytest

from agent.a2a_servers import disease_server, fertilize_server
from agent.a2a_servers.common import ExpertExecutor, build_a2a_app
from agent.agents.base import ExpertAnswer
from agent.agents.orchestrator.a2a_client import AgentRegistry
from agent.config import settings
from agent.state.agent_state import OrchestratorContext


class FakeExpert:
    def __init__(self, name="fertilize", framework="langchain", delay=0.0, degraded=False):
        self.name, self.framework, self.delay, self.degraded = name, framework, delay, degraded
        self.calls = []

    async def answer(self, question, user_id, image_path=None):
        self.calls.append({"question": question, "user_id": user_id, "image_path": image_path,
                           "image_existed": bool(image_path and Path(image_path).exists())})
        await asyncio.sleep(self.delay)
        return ExpertAnswer(text=f"Answer to: {question}", tools_used=["DosageCalculator"],
                            model="fake-model", degraded=self.degraded)


def make_registry(server_module, expert, headers=None):
    app = build_a2a_app(
        server_module.agent_card(), ExpertExecutor(lambda: expert, expert.name, expert.framework),
        readiness_checks=[lambda: (True, "fake")],
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers=headers or {}, timeout=10)
    return AgentRegistry([server_module.agent_card().supported_interfaces[0].url.rsplit("/a2a", 1)[0]], http), http


@pytest.fixture(autouse=True)
def no_auth_by_default(monkeypatch):
    monkeypatch.setattr(settings, "A2A_AUTH_TOKEN", "")
    monkeypatch.setattr(settings, "A2A_DISCOVERY_TTL_SECONDS", 0)


@pytest.mark.anyio
async def test_discovery_and_call_carry_metadata_both_ways():
    expert = FakeExpert()
    registry, _ = make_registry(fertilize_server, expert)

    [agent] = await registry.agents()
    assert agent.intent == "fertilize"
    assert agent.tool_name == "ask_tea_fertilizer_agent"
    assert not agent.accepts_images

    text, meta = await registry.call(agent, "How much fertilizer?", OrchestratorContext(user_id=7, trace_id="abc123"))
    assert text == "Answer to: How much fertilizer?"
    assert expert.calls[0]["user_id"] == 7  # arrived as an int, not a float
    assert meta["trace_id"] == "abc123"     # trace id went there and came back
    assert meta["framework"] == "langchain"
    assert meta["tools_used"] == ["DosageCalculator"]
    assert meta["status"] == "ok"


@pytest.mark.anyio
async def test_photo_is_forwarded_only_to_image_agents(tmp_path):
    photo = tmp_path / "leaf.jpg"
    photo.write_bytes(b"\xff\xd8\xff fake jpeg bytes")

    disease = FakeExpert(name="disease", framework="crewai")
    registry, _ = make_registry(disease_server, disease)
    [agent] = await registry.agents()
    assert agent.accepts_images
    await registry.call(agent, "What is this?", OrchestratorContext(user_id=3, image_path=str(photo)))
    assert disease.calls[0]["image_existed"]
    assert not Path(disease.calls[0]["image_path"]).exists()  # deleted after the answer

    fert = FakeExpert()
    registry, _ = make_registry(fertilize_server, fert)
    [agent] = await registry.agents()
    await registry.call(agent, "Dosage?", OrchestratorContext(user_id=3, image_path=str(photo)))
    assert fert.calls[0]["image_path"] is None


@pytest.mark.anyio
async def test_auth_token_is_required_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "A2A_AUTH_TOKEN", "s3cret")

    registry, http = make_registry(fertilize_server, FakeExpert(), headers={"Authorization": "Bearer s3cret"})
    [agent] = await registry.agents()
    assert "bearer" in agent.card.security_schemes  # the card advertises the requirement
    _, meta = await registry.call(agent, "Dosage?", OrchestratorContext(user_id=3))
    assert meta["status"] == "ok"

    anon = await http.post("http://localhost:9202/a2a", json={}, headers={"Authorization": "Bearer wrong"})
    assert anon.status_code == 401
    card = await http.get("http://localhost:9202/.well-known/agent-card.json", headers={"Authorization": ""})
    assert card.status_code == 200  # discovery stays public


@pytest.mark.anyio
async def test_invalid_user_id_is_rejected_by_expert():
    expert = FakeExpert()
    registry, _ = make_registry(fertilize_server, expert)
    [agent] = await registry.agents()
    # OrchestratorContext refuses user_id 0 itself; build the bad context to reach the server check.
    ctx = OrchestratorContext.model_construct(user_id=0, image_path=None, trace_id="t")
    _, meta = await registry.call(agent, "Dosage?", ctx)
    assert meta["status"] == "rejected"
    assert expert.calls == []  # never reached the LLM


@pytest.mark.anyio
async def test_expert_timeout_returns_degraded_reply(monkeypatch):
    monkeypatch.setattr(settings, "EXPERT_TIMEOUT_SECONDS", 0.05)
    registry, _ = make_registry(fertilize_server, FakeExpert(delay=1))
    [agent] = await registry.agents()
    text, meta = await registry.call(agent, "Dosage?", OrchestratorContext(user_id=3))
    assert meta["status"] == "timeout" and meta["degraded"]
    assert "too long" in text


@pytest.mark.anyio
async def test_repeated_failures_open_the_circuit(monkeypatch):
    monkeypatch.setattr(settings, "A2A_CONNECT_RETRIES", 0)
    registry, _ = make_registry(fertilize_server, FakeExpert(degraded=True))
    [agent] = await registry.agents()

    # Make the expert unreachable from now on.
    def refuse(request):
        raise httpx.ConnectError("connection refused")
    registry.http._transport = httpx.MockTransport(refuse)

    for _ in range(settings.CIRCUIT_FAILURE_THRESHOLD):
        text, meta = await registry.call(agent, "Dosage?", OrchestratorContext(user_id=3))
        assert meta["status"] == "error" and "temporarily unavailable" in text
    _, meta = await registry.call(agent, "Dosage?", OrchestratorContext(user_id=3))
    assert meta["status"] == "circuit_open"  # failed instantly, no network call


@pytest.mark.anyio
async def test_unreachable_agent_is_skipped_then_found_later():
    expert = FakeExpert()
    app = build_a2a_app(fertilize_server.agent_card(), ExpertExecutor(lambda: expert, "fertilize", "langchain"), [])
    up = {"value": False}
    asgi = httpx.ASGITransport(app=app)

    class Flaky(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if not up["value"]:
                raise httpx.ConnectError("not started yet")
            return await asgi.handle_async_request(request)

    registry = AgentRegistry(["http://localhost:9202"], httpx.AsyncClient(transport=Flaky()))
    assert await registry.agents() == []
    up["value"] = True
    assert len(await registry.agents()) == 1  # TTL is 0 in tests, so it looks again


@pytest.mark.anyio
async def test_health_ready_and_metrics_endpoints():
    app = build_a2a_app(
        fertilize_server.agent_card(), ExpertExecutor(FakeExpert, "fertilize", "langchain"),
        readiness_checks=[lambda: (False, "no key")],
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        assert (await http.get("/healthz")).status_code == 200
        ready = await http.get("/readyz")
        assert ready.status_code == 503 and ready.json()["ready"] is False
        metrics = await http.get("/metrics/")
        assert metrics.status_code == 200 and "a2a_expert_requests_total" in metrics.text


def test_production_refuses_to_start_without_token(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    with pytest.raises(RuntimeError, match="A2A_AUTH_TOKEN"):
        build_a2a_app(fertilize_server.agent_card(), ExpertExecutor(FakeExpert, "f", "langchain"), [])
