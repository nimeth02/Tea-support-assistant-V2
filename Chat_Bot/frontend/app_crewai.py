"""UI for the CrewAI crew variant (Chat_Bot/agent (crewAI)).

How this variant works: a "crew" of role-playing agents. A manager agent (the coordinator)
reads the question and delegates parts of it to expert crew members, then writes the final
answer. This UI shows the crew roster and who the manager delegated to.

Run:  python launch.py up crewai      (UI on http://localhost:8513)
"""
import streamlit as st

from ui_common import UI, expert_badges, experts_used

ui = UI("crewai")
ui.page_setup("👥")

EXAMPLES = [
    "My tea leaves have brown spots, what should I do?",
    "How much fertilizer should I use, and will it rain this week?",
    "When was my last fertilization?",
    "Which dates does the company not collect tea leaves?",
]

MANAGER = "Tea support coordinator"
# crew member role (from the agent files) -> intent label its tools report
CREW = {
    "Tea disease and pest expert": ("disease", "🦠"),
    "Tea fertilizer expert": ("fertilize", "🌱"),
    "Tea company help desk": ("company_help", "🏢"),
}


def crew_dot(used=()) -> str:
    color = ui.variant.color
    lines = []
    for i, (role, (intent, icon)) in enumerate(CREW.items()):
        on = intent in used
        fill = f'fillcolor="{color}", fontcolor="white"' if on else 'fillcolor="#fdeee3"'
        lines.append(f'm{i} [label="{icon} {role}", {fill}];')
        attrs = f'color="{color}", penwidth=3, label="delegated"' if on else 'color="#9e9e9e", style=dashed'
        lines.append(f"manager -> m{i} [{attrs}];")
    body = "\n".join(lines)
    return f"""digraph G {{
        rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=9];
        task [label="📋 Task: answer the farmer", shape=note, fillcolor="#eeeeee"];
        manager [label="👔 {MANAGER}\\n(manager, allow_delegation)", fillcolor="#fbd3b5"];
        task -> manager [color="#9e9e9e"];
        {body}
    }}"""


def intro():
    ui.banner("👥", "CrewAI hierarchical crew · a manager agent delegates to expert crew members")
    last = st.session_state.get("last_reply")
    used = experts_used(last.get("intent")) if last else []
    with st.expander("How this version works", expanded=not st.session_state.get("messages")):
        st.markdown(
            "**CrewAI** organises agents like a team. Each agent has a **role**, a **goal** and a "
            "backstory. This crew is **hierarchical**: a manager (the coordinator) gets the task, "
            "decides which expert to hand work to (\"delegation\"), checks their answers, and "
            "writes the final reply. If a model fails, the whole crew re-runs on the next fallback model.\n\n"
            "- ✅ Easy to read and extend: add an agent = add a role and its tools.\n"
            "- ⚠️ The slowest version: the manager reasons step by step, so it makes the most LLM calls.\n\n"
            "Orange arrows = crew members the manager delegated to for your **last** question. "
            "CrewAI doesn't speak LangSmith natively; this project bridges CrewAI's events into "
            "LangSmith, so each crew run shows up as a trace of tasks, agents, LLM calls and tools."
        )
        st.graphviz_chart(crew_dot(used))


def details(data: dict):
    expert_badges(data.get("intent"), prefix="**Manager delegated to:**")


def sidebar(info):
    st.subheader("👥 Crew roster")
    last = st.session_state.get("last_reply")
    used = experts_used(last.get("intent")) if last else []
    st.markdown(f"👔 **{MANAGER}** · manager")
    for role, (intent, icon) in CREW.items():
        mark = "🟧" if intent in used else "⬜"
        st.markdown(f"{mark} {icon} {role}")
    st.caption("🟧 = worked on the last question")


ui.run(lambda: ui.chat_page(EXAMPLES, intro, details,
                            "The crew is working… (the manager delegates step by step - can take 1-2 min)"),
       sidebar)
