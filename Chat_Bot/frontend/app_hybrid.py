"""UI for the multi-framework A2A variant (Chat_Bot/agent (crew AI , langchain)).

How this variant works: each expert is built with a different agent framework (disease: CrewAI,
fertilizer: LangChain, company help: LangGraph) and runs as its own A2A server. A LangChain
supervisor reads their Agent Cards and messages them over HTTP. This UI also shows the AIOps
side: each expert's readiness checks and the orchestrator's circuit breakers.

Run:  python launch.py up hybrid      (UI on http://localhost:8514; also starts the 3 experts)
"""
import re

import requests
import streamlit as st

from ui_common import UI, expert_badges, experts_used

ui = UI("hybrid")
ui.page_setup("🧩")

EXAMPLES = [
    "My tea leaves have small pale blisters, what is it?",
    "How much fertilizer should I use, and will it rain this week?",
    "How can I get an advance on my leaf payment?",
    "My leaves have spots - also, when did I last fertilize?",
]

DEFAULT_URLS = ["http://localhost:9201", "http://localhost:9202", "http://localhost:9203"]
CARD_PATH = "/.well-known/agent-card.json"
FRAMEWORK_ICONS = {"crewai": "🚣 CrewAI", "langchain": "🦜 LangChain", "langgraph": "🕸️ LangGraph"}
CIRCUIT = {0: "🟢 closed", 1: "🟡 half-open", 2: "🔴 open"}


@st.cache_data(ttl=10, show_spinner=False)
def fetch(url: str, path: str):
    """JSON from an expert (card or readiness), or None if its server is not reachable."""
    try:
        resp = requests.get(url + path, timeout=3)
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


@st.cache_data(ttl=5, show_spinner=False)
def circuit_states(api_url: str) -> dict:
    """Circuit breaker per expert, read from the backend's Prometheus metrics."""
    try:
        text = requests.get(f"{api_url}/metrics", timeout=3).text
    except requests.RequestException:
        return {}
    return {m.group(1): int(float(m.group(2)))
            for m in re.finditer(r'a2a_circuit_state\{agent="([^"]+)"\} ([0-9.]+)', text)}


def framework_of(card: dict) -> str:
    tags = (card.get("skills") or [{}])[0].get("tags", [])
    return next((t for t in tags if t in FRAMEWORK_ICONS), "?")


def intent_of(card: dict) -> str:
    return (card.get("skills") or [{}])[0].get("id", "")


def agent_urls(info) -> list:
    return (info or {}).get("a2a_agent_urls") or DEFAULT_URLS


def network_dot(urls, used=()) -> str:
    color = ui.variant.color
    lines = []
    for i, url in enumerate(urls):
        card = fetch(url, CARD_PATH)
        port = url.rsplit(":", 1)[-1]
        if card:
            on = intent_of(card) in used
            fill = f'fillcolor="{color}", fontcolor="white"' if on else 'fillcolor="#e0f2f1"'
            label = f'{card.get("name", "agent")}\\n{FRAMEWORK_ICONS.get(framework_of(card), "")} :{port}'
        else:
            on, fill, label = False, 'fillcolor="#fde0e0"', f"offline\\n:{port}"
        lines.append(f'a{i} [label="{label}", {fill}];')
        attrs = f'color="{color}", penwidth=3' if on else 'color="#9e9e9e", style=dashed'
        lines.append(f'orch -> a{i} [label="A2A", {attrs}];')
    body = "\n".join(lines)
    return f"""digraph G {{
        rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=9];
        orch [label="🧩 LangChain supervisor\\n(backend :{ui.variant.backend_port})", fillcolor="#b2dfdb"];
        {body}
    }}"""


def make_intro(info):
    def intro():
        ui.banner("🧩", "One question, three frameworks: CrewAI, LangChain and LangGraph experts talking over A2A")
        last = st.session_state.get("last_reply")
        used = experts_used(last.get("intent")) if last else []
        with st.expander("How this version works", expanded=not st.session_state.get("messages")):
            st.markdown(
                "Each expert is written with a **different agent framework**, and each runs as its "
                "own **A2A server** (Agent2Agent protocol). The supervisor only sees their **Agent "
                "Cards**, so it doesn't know or care which framework is inside.\n\n"
                "- 🚣 **CrewAI** disease expert: a one-agent crew with the leaf-photo model\n"
                "- 🦜 **LangChain** fertilizer expert: a tool-calling agent over the farm database\n"
                "- 🕸️ **LangGraph** help desk: a fixed retrieve → answer graph\n\n"
                "Production extras: shared-token auth between agents, retries and circuit breakers, "
                "readiness checks, Prometheus metrics, JSON logs with one trace id across every agent, "
                "guardrails, and an eval suite that gates releases."
            )
            st.graphviz_chart(network_dot(agent_urls(info), used))
    return intro


def details(data: dict):
    expert_badges(data.get("intent"), prefix="**Experts messaged over A2A:**")


def sidebar(info):
    st.subheader("🌐 Experts")
    circuits = circuit_states(ui.api_url)
    last = st.session_state.get("last_reply")
    used = experts_used(last.get("intent")) if last else []
    for url in agent_urls(info):
        card = fetch(url, CARD_PATH)
        if not card:
            st.markdown(f"🔴 **offline** · `{url}`")
            continue
        ready = fetch(url, "/readyz") or {}
        mark = "🟦" if intent_of(card) in used else ("🟢" if ready.get("ready") else "🟠")
        with st.expander(f"{mark} {card.get('name', url)} · {FRAMEWORK_ICONS.get(framework_of(card), '')}"):
            st.caption(card.get("description", ""))
            circuit = circuits.get(intent_of(card))
            st.markdown(f"**Circuit breaker:** {CIRCUIT.get(circuit, 'not used yet')}")
            for name, check in (ready.get("checks") or {}).items():
                st.markdown(f"{'✅' if check['ok'] else '❌'} `{name}` · {check['detail']}")
            st.json(card, expanded=False)
    if st.button("↻ Re-check experts", width="stretch"):
        fetch.clear()
        circuit_states.clear()
        st.rerun()
    st.caption("🟢 ready · 🟠 up but not ready · 🟦 messaged for the last question · 🔴 offline")


info = ui.backend_info()
ui.run(lambda: ui.chat_page(EXAMPLES, make_intro(info), details,
                            "Messaging the experts over A2A… (can take ~30-60s on free-tier Gemini)"),
       sidebar)
