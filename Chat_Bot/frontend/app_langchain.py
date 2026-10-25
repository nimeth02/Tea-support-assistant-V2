"""UI for the LangChain supervisor variant (Chat_Bot/agent (langchain)).

How this variant works: there is no fixed route. One "supervisor" LLM has each expert agent as
a *tool* it may call - none, one, or several - and then writes one combined answer. This UI
shows which experts the supervisor decided to consult.

Run:  python launch.py up langchain      (UI on http://localhost:8511)
"""
import streamlit as st

from ui_common import UI, expert_badges, experts_used

ui = UI("langchain")
ui.page_setup("🧭")

EXAMPLES = [
    "My tea leaves have brown spots, and how much fertilizer should I use?",
    "Will it rain on my tea land this week?",
    "When was my last fertilization?",
    "Which dates does the company not collect tea leaves?",
    "What's the capital of France?",  # off-topic: the supervisor should call no expert
]

# expert tool name -> (intent label it reports, what it covers)
TOOLS = {
    "ask_disease_expert": ("disease", "Diseases, pests, leaf photos"),
    "ask_fertilizer_expert": ("fertilize", "Fertilizer dosage, history, weather, land"),
    "ask_company_help": ("company_help", "Collection days, payments, policies"),
}


def supervisor_dot(used=()) -> str:
    color = ui.variant.color
    lines = []
    for tool, (intent, _) in TOOLS.items():
        on = intent in used
        fill = f'fillcolor="{color}", fontcolor="white"' if on else 'fillcolor="#eef3fb"'
        lines.append(f'{tool} [label="🔧 {tool}", {fill}];')
        attrs = f'color="{color}", penwidth=3, dir=both' if on else 'color="#9e9e9e", style=dashed, dir=both'
        lines.append(f"supervisor -> {tool} [{attrs}];")
    body = "\n".join(lines)
    return f"""digraph G {{
        rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
        farmer [label="👩‍🌾 Farmer", shape=ellipse, fillcolor="#e0e0e0"];
        supervisor [label="🧭 Supervisor LLM\\n(decides which tools,\\nloops up to 6 times)", fillcolor="#d6e4f7"];
        farmer -> supervisor [dir=both, color="#9e9e9e"];
        {body}
    }}"""


def intro():
    ui.banner("🧭", "LangChain create_agent supervisor · the LLM chooses which experts to consult")
    last = st.session_state.get("last_reply")
    used = experts_used(last.get("intent")) if last else []
    with st.expander("How this version works", expanded=not st.session_state.get("messages")):
        st.markdown(
            "Here a **supervisor** LLM is in charge. Each expert agent is offered to it as a "
            "**tool** (a function the LLM can decide to call). The supervisor can call **several** "
            "experts for one question, read their answers, and then write a single reply.\n\n"
            "- ✅ Handles mixed questions (\"brown spots *and* how much fertilizer?\").\n"
            "- ⚠️ Less predictable and uses more LLM calls than a fixed graph.\n\n"
            "Blue arrows = tools the supervisor called for your **last** question. Try the first "
            "example in the sidebar - it needs two experts."
        )
        st.graphviz_chart(supervisor_dot(used))


def details(data: dict):
    used = experts_used(data.get("intent"))
    n = len([u for u in used if u not in ("unknown", "error")])
    expert_badges(data.get("intent"), prefix=f"**Experts consulted ({n}):**")


def sidebar(info):
    st.subheader("🔧 Supervisor's tools")
    last = st.session_state.get("last_reply")
    used = experts_used(last.get("intent")) if last else []
    for tool, (intent, desc) in TOOLS.items():
        mark = "🟦" if intent in used else "⬜"
        st.markdown(f"{mark} `{tool}`  \n<small>{desc}</small>", unsafe_allow_html=True)
    st.caption("🟦 = called for the last question")


ui.run(lambda: ui.chat_page(EXAMPLES, intro, details,
                            "Supervisor is consulting experts… (can take ~30-60s on free-tier Gemini)"),
       sidebar)
