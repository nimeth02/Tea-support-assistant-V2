"""UI for the LangGraph router variant (Chat_Bot/agent).

How this variant works: a fixed graph. The first node classifies the question into ONE intent,
then the graph routes it to exactly one expert node. This UI draws that graph and lights up the
path the last question took.

Run:  python launch.py up agent      (UI on http://localhost:8510)
"""
import streamlit as st

from ui_common import UI, INTENT_LABELS

ui = UI("agent")
ui.page_setup("🍃")

EXAMPLES = [
    "When was my last fertilization?",
    "How much fertilizer should I use?",
    "My tea leaves have brown spots, what should I do?",
    "Which dates does the company not collect tea leaves?",
]

# intent -> the graph node it routes to (mirrors route_by_intent in orchestrator_agent.py)
ROUTES = {"disease": "disease_agent", "fertilize": "fertilize_agent",
          "company_help": "help_agent", "help": "help_agent"}


def route_of(data: dict) -> str:
    if data.get("has_image"):
        return "disease_agent"  # a photo always goes to the disease node
    return ROUTES.get(data.get("intent"), "END")


def graph_dot(active: str = None) -> str:
    """The LangGraph workflow as a Graphviz diagram; the active route is highlighted."""
    color = ui.variant.color

    def node(name, label):
        on = name == active or (name == "classify_intent" and active)
        style = f'style="filled", fillcolor="{color}", fontcolor="white"' if on else 'style="filled", fillcolor="#f1f3f1"'
        return f'{name} [label="{label}", {style}];'

    def edge(a, b, label=""):
        on = active and b == active
        attrs = f'color="{color}", penwidth=3' if on else 'color="#9e9e9e"'
        return f'{a} -> {b} [label="{label}", {attrs}];'

    return f"""digraph G {{
        rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=9];
        START [shape=circle, label="start", fillcolor="#e0e0e0"];
        {node("classify_intent", "classify_intent\\n(LLM picks 1 intent)")}
        {node("disease_agent", "disease_agent\\n🦠 RAG + web + photo")}
        {node("fertilize_agent", "fertilize_agent\\n🌱 dosage, weather, history")}
        {node("help_agent", "help_agent\\n🏢 company RAG")}
        {node("END", "END\\n(no expert)")}
        START -> classify_intent [color="#9e9e9e"];
        {edge("classify_intent", "disease_agent", "disease / photo")}
        {edge("classify_intent", "fertilize_agent", "fertilize")}
        {edge("classify_intent", "help_agent", "company_help")}
        {edge("classify_intent", "END", "unknown")}
    }}"""


def intro():
    ui.banner("🕸️", "LangGraph state graph · one classifier step, then exactly one expert per question")
    last = st.session_state.get("last_reply")
    with st.expander("How this version works", expanded=not st.session_state.get("messages")):
        st.markdown(
            "This version is a **fixed workflow** (a *graph*: boxes are steps, arrows are the "
            "allowed paths). First an LLM reads your question and picks **one** label - disease, "
            "fertilize or company help. Then the graph sends the question to that one expert.\n\n"
            "- ✅ Predictable: the same kind of question always takes the same path.\n"
            "- ⚠️ A mixed question (disease *and* fertilizer) only reaches one expert.\n\n"
            "The highlighted path is the route your **last** question took."
        )
        st.graphviz_chart(graph_dot(route_of(last) if last else None))


def details(data: dict):
    route = route_of(data)
    label, color = INTENT_LABELS.get(data.get("intent") or "unknown", (data.get("intent"), "gray"))
    st.markdown(f":gray-badge[route] `classify_intent` → `{route}` &nbsp; :{color}-badge[{label}]")


def sidebar(info):
    st.subheader("🕸️ Graph nodes")
    last = st.session_state.get("last_reply")
    active = route_of(last) if last else None
    for name in ["classify_intent", "disease_agent", "fertilize_agent", "help_agent"]:
        mark = "🟩" if active and (name == active or name == "classify_intent") else "⬜"
        st.markdown(f"{mark} `{name}`")
    st.caption("🟩 = visited by the last question")


ui.run(lambda: ui.chat_page(EXAMPLES, intro, details,
                            "Classifying and routing… (free-tier Gemini is rate limited, this can take ~30s)"),
       sidebar)
