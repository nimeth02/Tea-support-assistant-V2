"""UI for the A2A agent-network variant (Chat_Bot/agent A2A).

How this variant works: each expert is its own little web server (an "A2A agent") that
describes itself in an *Agent Card*. The supervisor reads those cards, then sends questions to
the agents over HTTP. This UI shows the live network: which agent servers are up, what their
cards say, and which ones were messaged for the last question.

Run:  python launch.py up a2a      (UI on http://localhost:8512; also starts the 3 agent servers)
"""
import requests
import streamlit as st

from ui_common import UI, expert_badges, experts_used

ui = UI("a2a")
ui.page_setup("🛰️")

EXAMPLES = [
    "My tea leaves have small pale blisters, what is it?",
    "How much fertilizer should I use, and will it rain this week?",
    "Which dates does the company not collect tea leaves?",
    "My leaves have spots - also, when do you pay suppliers?",
]

DEFAULT_URLS = ["http://localhost:9101", "http://localhost:9102", "http://localhost:9103"]
CARD_PATH = "/.well-known/agent-card.json"


@st.cache_data(ttl=10, show_spinner=False)
def fetch_card(url: str):
    """The agent's A2A Agent Card (JSON), or None if its server is not reachable."""
    try:
        resp = requests.get(url + CARD_PATH, timeout=2)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def card_intent(card: dict) -> str:
    """By project convention the first skill's id is the agent's intent label."""
    skills = card.get("skills") or []
    return skills[0].get("id") if skills else ""


def agent_urls(info) -> list:
    return (info or {}).get("a2a_agent_urls") or DEFAULT_URLS


def network_dot(urls, used=()) -> str:
    color = ui.variant.color
    lines = []
    for i, url in enumerate(urls):
        card = fetch_card(url)
        port = url.rsplit(":", 1)[-1]
        if card:
            on = card_intent(card) in used
            fill = f'fillcolor="{color}", fontcolor="white"' if on else 'fillcolor="#f3e9f7"'
            label = f'{card.get("name", "agent")}\\n:{port}  🟢'
        else:
            on, fill, label = False, 'fillcolor="#fde0e0"', f"offline\\n:{port}  🔴"
        lines.append(f'a{i} [label="{label}", {fill}];')
        attrs = f'color="{color}", penwidth=3' if on else 'color="#9e9e9e", style=dashed'
        lines.append(f'orch -> a{i} [label="JSON-RPC", {attrs}];')
    body = "\n".join(lines)
    return f"""digraph G {{
        rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=9];
        orch [label="🛰️ Orchestrator\\n(backend :{ui.variant.backend_port})\\nreads Agent Cards", fillcolor="#e7d6f0"];
        {body}
    }}"""


def make_intro(info):
    def intro():
        ui.banner("🛰️", "Agent2Agent (A2A) protocol · experts are separate servers the orchestrator messages over HTTP")
        last = st.session_state.get("last_reply")
        used = experts_used(last.get("intent")) if last else []
        with st.expander("How this version works", expanded=not st.session_state.get("messages")):
            st.markdown(
                "**A2A** (Agent2Agent) is an open protocol for agents to talk to each other over "
                "the network, whatever framework each one is built with. Here every expert runs as "
                "its **own web server**. It publishes an **Agent Card** (a small JSON file saying who "
                "it is and what it can do). The orchestrator reads the cards at start-up, then "
                "messages the agents it needs.\n\n"
                "- ✅ Agents can be added, moved to other machines, or rewritten in other frameworks.\n"
                "- ⚠️ Every expert call is a network hop, so there are more moving parts to keep running.\n\n"
                "Purple arrows = agents messaged for your **last** question. In LangSmith, the remote "
                "agents' steps appear *inside* the orchestrator's trace (distributed tracing)."
            )
            st.graphviz_chart(network_dot(agent_urls(info), used))
    return intro


def details(data: dict):
    expert_badges(data.get("intent"), prefix="**Agents messaged over A2A:**")


def sidebar(info):
    st.subheader("🌐 Agent network")
    last = st.session_state.get("last_reply")
    used = experts_used(last.get("intent")) if last else []
    for url in agent_urls(info):
        card = fetch_card(url)
        if not card:
            st.markdown(f"🔴 **offline** · `{url}`")
            continue
        mark = "🟪" if card_intent(card) in used else "🟢"
        with st.expander(f"{mark} {card.get('name', url)}"):
            st.caption(card.get("description", ""))
            st.markdown(f"**URL:** `{url}`  \n**Accepts:** {', '.join(card.get('defaultInputModes') or card.get('default_input_modes') or [])}")
            for skill in card.get("skills", []):
                st.markdown(f"- **{skill.get('name')}** (`{skill.get('id')}`)")
            st.json(card, expanded=False)
    if st.button("↻ Re-check agents", width="stretch"):
        fetch_card.clear()
        st.rerun()
    st.caption("🟢 online · 🟪 messaged for the last question · 🔴 offline "
               "(start with `python launch.py a2a-servers`)")


info = ui.backend_info()
ui.run(lambda: ui.chat_page(EXAMPLES, make_intro(info), details,
                            "Messaging A2A agents… (can take ~30-60s on free-tier Gemini)"),
       sidebar)
