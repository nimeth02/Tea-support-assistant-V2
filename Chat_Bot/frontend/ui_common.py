"""Building blocks shared by the Streamlit UIs (app_agent.py, app_langchain.py, app_a2a.py,
app_crewai.py, app_hybrid.py, app_voice.py).

Each app file only holds what is different about its variant - the intro, the diagram, what
it shows under an answer and in the sidebar, and (for the voice variant) a microphone next to
the paperclip. This module holds the rest: talking to the backend, the chat loop, the LangSmith
links and the Observability dashboard.
"""
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Chat_Bot/, for variants.py
from variants import VARIANTS, Variant  # noqa: E402

CHAT_TIMEOUT = 300  # agents are rate limited, so a single chat can take a while

INTENT_LABELS = {
    "disease": ("🦠 Disease expert", "red"),
    "fertilize": ("🌱 Fertilizer expert", "green"),
    "company_help": ("🏢 Company help", "blue"),
    "help": ("🏢 Company help", "blue"),
    "unknown": ("❔ No expert needed", "gray"),
    "error": ("⚠️ Error", "orange"),
}


class UI:
    """One variant's UI session: knows its backend URL and draws the shared parts."""

    def __init__(self, key: str):
        self.variant: Variant = VARIANTS[key]
        self.api_url = os.getenv("TEA_API_URL", self.variant.backend_url)

    # ------------------------------------------------------------ API helpers
    def api_get(self, path: str, timeout: float = 15, **params):
        resp = requests.get(f"{self.api_url}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def backend_info(self) -> Optional[dict]:
        """The backend's /variant info, or None if it is offline."""
        try:
            return self.api_get("/variant", timeout=5)
        except requests.RequestException:
            return None

    def send_chat(self, message: str, image_file=None, audio_file=None, **fields) -> dict:
        """POST /chat. `fields` are extra form fields (e.g. language="si", speak=True)."""
        files = {}
        if image_file:
            files["image"] = (image_file.name, image_file.getvalue(), image_file.type)
        if audio_file:
            files["audio"] = (audio_file.name or "voice.wav", audio_file.getvalue(),
                              audio_file.type or "audio/wav")
        data = {"message": message}
        for key, value in fields.items():
            if value is not None:
                data[key] = str(value).lower() if isinstance(value, bool) else value
        resp = requests.post(f"{self.api_url}/chat", data=data, files=files or None,
                             timeout=CHAT_TIMEOUT)
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text) if resp.headers.get(
                "content-type", "").startswith("application/json") else resp.text
            return {"response": f"⚠️ Backend error ({resp.status_code}): {detail}",
                    "intent": "error", "trace_id": resp.headers.get("X-Trace-Id")}
        return resp.json()

    # ------------------------------------------------------------ Look and feel
    def page_setup(self, icon: str) -> None:
        st.set_page_config(page_title=f"Tea Assistant · {self.variant.title}", page_icon=icon, layout="wide")
        c = self.variant.color
        st.markdown(f"""<style>
            .variant-banner {{ background: linear-gradient(90deg, {c} 0%, {c}cc 60%, {c}55 100%);
                color: white; padding: 0.9rem 1.2rem; border-radius: 0.6rem; margin-bottom: 0.8rem; }}
            .variant-banner h2 {{ color: white; margin: 0; padding: 0; font-size: 1.6rem; }}
            .variant-banner p {{ margin: 0.2rem 0 0 0; opacity: 0.92; }}
            section[data-testid="stSidebar"] {{ border-right: 3px solid {c}; }}
        </style>""", unsafe_allow_html=True)

    def banner(self, icon: str, subtitle: str) -> None:
        st.markdown(f"""<div class="variant-banner"><h2>{icon} Tea Support Assistant · {self.variant.title}</h2>
            <p>{subtitle}</p></div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------ Sidebar
    def sidebar_status(self, info: Optional[dict]) -> None:
        if info:
            st.success(f"Backend online · {self.api_url}", icon="🟢")
            if info.get("key") != self.variant.key:
                st.warning(f"This backend runs the **{info.get('title')}** variant, not "
                           f"{self.variant.title}. Start the right one with "
                           f"`python launch.py up {self.variant.key}`.")
        else:
            st.error(f"Backend offline · {self.api_url}", icon="🔴")
            st.caption(f"Start it with `python launch.py up {self.variant.key}`")

    def sidebar_langsmith(self, info: Optional[dict]) -> None:
        st.subheader("🛰️ LangSmith")
        project = (info or {}).get("langsmith_project", self.variant.langsmith_project)
        if info and not info.get("langsmith_enabled"):
            st.caption(f"Tracing is **off**. Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` "
                       f"in `.env`, then restart. Traces will go to project `{project}`.")
            return
        st.caption(f"Project: `{project}`")
        url = (info or {}).get("langsmith_project_url")
        if url:
            st.link_button("Open project in LangSmith", url, width="stretch")
        else:
            st.caption("The project appears in LangSmith after the first chat.")

    # ------------------------------------------------------------ Chat page
    def chat_page(self, examples: List[str], intro: Callable[[], None],
                  details: Callable[[dict], None], spinner: str,
                  composer: Callable[[str], object] = None,
                  request_fields: Callable[[], dict] = None) -> None:
        """The chat loop. `intro` draws the top of the page; `details(data)` draws the
        variant-specific part under each answer (data = the backend's JSON reply).

        Optional hooks for variants with more ways to ask:
          composer(widget_key)  draws an extra input next to the paperclip (the voice variant's
                                microphone) and returns a recorded audio file, or None
          request_fields()      extra form fields to send with every message (e.g. language)
        """
        intro()

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        if "composer_key" not in st.session_state:
            st.session_state.composer_key = 0

        with st.sidebar:
            st.subheader("Try asking")
            for example in examples:
                if st.button(example, width="stretch"):
                    st.session_state.pending = example
            if st.button("🗑️ Clear chat", width="stretch"):
                st.session_state.messages = []
                st.rerun()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("data"):
                    self._answer_footer(msg["data"], details)

        # ChatGPT-style attachment: a paperclip that opens the uploader; the chosen file shows as a chip.
        # A variant with a `composer` (the voice variant's microphone) gets a column for it too.
        if composer:
            attach_col, composer_col, chip_col = st.columns([0.07, 0.38, 0.55])
        else:
            attach_col, chip_col = st.columns([0.07, 0.93])
            composer_col = None
        with attach_col:
            with st.popover("📎", help="Attach a tea leaf photo"):
                uploaded_image = st.file_uploader(
                    "Tea leaf photo for disease identification",
                    type=["jpg", "jpeg", "png"],
                    key=f"uploader_{st.session_state.uploader_key}",
                )
        recorded_audio = None
        if composer_col is not None:
            with composer_col:
                recorded_audio = composer(f"composer_{st.session_state.composer_key}")
        with chip_col:
            if uploaded_image:
                thumb_col, name_col, remove_col = st.columns([0.1, 0.8, 0.1])
                with thumb_col:
                    st.image(uploaded_image, width=36)
                with name_col:
                    st.caption(f"📎 {uploaded_image.name}")
                with remove_col:
                    if st.button("✕", key="remove_attachment", help="Remove attachment"):
                        st.session_state.uploader_key += 1
                        st.rerun()

        typed = st.chat_input("Type your question…") or st.session_state.pop("pending", None)
        if not typed and not uploaded_image and not recorded_audio:
            return
        # A photo on its own asks the default disease question. A voice clip on its own is sent
        # with an empty message - the backend turns the speech into text.
        prompt = typed or ("" if recorded_audio else "What disease or pest is affecting this tea leaf?")

        if uploaded_image:
            st.session_state.uploader_key += 1  # clear the attachment now that it's being sent
        if recorded_audio:
            st.session_state.composer_key += 1  # clear the recording too

        display_text = (prompt or "🎤 Voice message") + (" 🎤" if typed and recorded_audio else "") \
            + (" 📷" if uploaded_image else "")
        st.session_state.messages.append({"role": "user", "content": display_text})
        with st.chat_message("user"):
            st.markdown(display_text)

        with st.chat_message("assistant"):
            with st.spinner(spinner):
                try:
                    data = self.send_chat(prompt, uploaded_image, recorded_audio,
                                          **(request_fields() if request_fields else {}))
                except requests.RequestException as e:
                    data = {"response": f"⚠️ Could not reach the backend at {self.api_url}: {e}",
                            "intent": "error"}
            data["has_image"] = bool(uploaded_image)
            if data.get("transcript") and not typed:
                # Replace the "Voice message" placeholder with the words the backend heard.
                st.session_state.messages[-1]["content"] = f"🎤 {data['transcript']}" + (" 📷" if uploaded_image else "")
            answer = data.get("response") or "_No answer — the model may be rate limited, try again in a minute._"
            st.markdown(answer)
            self._answer_footer(data, details)
        st.session_state.messages.append({"role": "assistant", "content": answer, "data": data})
        st.session_state.last_reply = data
        st.rerun()  # redraw so the sidebar/diagram highlight this reply's route, not the previous one

    def _answer_footer(self, data: dict, details: Callable[[dict], None]) -> None:
        details(data)
        meta = " · ".join(filter(None, [
            f"latency: {fmt_ms(data['latency_ms'])}" if data.get("latency_ms") else None,
            f"trace: {data['trace_id']}" if data.get("trace_id") else None,
        ]))
        cols = st.columns([0.75, 0.25])
        cols[0].caption(meta)
        if data.get("langsmith_url"):
            cols[1].link_button("🛰️ View in LangSmith", data["langsmith_url"])

    # ------------------------------------------------------------ Observability page
    def observability_page(self) -> None:
        head, refresh = st.columns([6, 1])
        head.title(f"📈 Observability · {self.variant.title}")
        refresh.button("🔄 Refresh", width="stretch")
        st.caption("Local traces recorded by this variant's backend. For the full step-by-step "
                   "trace tree, use the LangSmith link in the sidebar.")

        try:
            stats = self.api_get("/observability/stats")
            traces = pd.DataFrame(self.api_get("/observability/traces", limit=500))
            spans = pd.DataFrame(self.api_get("/observability/spans", limit=5000))
        except requests.RequestException as e:
            st.error(f"Could not load observability data from {self.api_url}: {e}")
            return

        total = stats["total_requests"]
        error_rate = (stats["errors"] / total * 100) if total else 0
        c = st.columns(6)
        c[0].metric("Chat requests", total)
        c[1].metric("Error rate", f"{error_rate:.0f}%")
        c[2].metric("p50 latency", fmt_ms(stats["p50_latency_ms"]))
        c[3].metric("p95 latency", fmt_ms(stats["p95_latency_ms"]))
        c[4].metric("LLM calls", stats["llm_calls"], delta=f"{stats['llm_errors']} errors",
                    delta_color="inverse" if stats["llm_errors"] else "off")
        c[5].metric("Tokens", f"{stats['input_tokens'] + stats['output_tokens']:,}",
                    help=f"{stats['input_tokens']:,} in / {stats['output_tokens']:,} out")

        if traces.empty:
            st.info("No traces yet — send a message on the Chat page.")
            return

        traces["time"] = pd.to_datetime(traces["ts"], unit="s")
        traces["latency_s"] = traces["latency_ms"] / 1000

        left, right = st.columns(2)
        with left:
            st.subheader("Chat latency")
            st.altair_chart(
                alt.Chart(traces).mark_circle(size=70).encode(
                    x=alt.X("time:T", title=None),
                    y=alt.Y("latency_s:Q", title="seconds"),
                    color=alt.Color("status:N", scale=alt.Scale(domain=["ok", "error"],
                                                                 range=["#2e7d32", "#c62828"])),
                    tooltip=["trace_id", "intent", "status", alt.Tooltip("latency_s:Q", format=".1f"),
                             "message"],
                ).properties(height=260),
                width="stretch",
            )
        with right:
            st.subheader("Requests by intent")
            intents = pd.DataFrame(stats["intents"].items(), columns=["intent", "count"])
            st.altair_chart(
                alt.Chart(intents).mark_bar(color=self.variant.color).encode(
                    x=alt.X("count:Q", title="requests"),
                    y=alt.Y("intent:N", sort="-x", title=None),
                    tooltip=["intent", "count"],
                ).properties(height=260),
                width="stretch",
            )

        if not spans.empty:
            left, right = st.columns(2)
            tools = spans[spans["kind"] == "tool"]
            llms = spans[spans["kind"] == "llm"]
            with left:
                st.subheader("Tool usage")
                if tools.empty:
                    st.caption("No tool calls yet.")
                else:
                    summary = tools.groupby("name").agg(
                        calls=("span_id", "count"),
                        errors=("status", lambda s: (s != "ok").sum()),
                        avg_latency=("latency_ms", "mean"),
                    ).reset_index()
                    summary["avg_latency"] = summary["avg_latency"].map(fmt_ms)
                    st.dataframe(summary, hide_index=True, width="stretch")
            with right:
                st.subheader("LLM calls")
                if llms.empty:
                    st.caption("No LLM calls yet.")
                else:
                    summary = llms.groupby(["model", "status"]).agg(
                        calls=("span_id", "count"),
                        avg_latency=("latency_ms", "mean"),
                        tokens_in=("input_tokens", "sum"),
                        tokens_out=("output_tokens", "sum"),
                    ).reset_index()
                    summary["avg_latency"] = summary["avg_latency"].map(fmt_ms)
                    st.dataframe(summary, hide_index=True, width="stretch")
        elif self.variant.key == "crewai":
            st.caption("CrewAI doesn't send LangChain callbacks, so no local LLM/tool spans are "
                       "recorded here - see the step-by-step crew trace in LangSmith instead.")

        st.subheader("Traces")
        table = traces[["time", "trace_id", "intent", "status", "latency_ms", "llm_calls",
                        "tool_calls", "total_tokens", "message"]].copy()
        table["latency_ms"] = table["latency_ms"].map(fmt_ms)
        selection = st.dataframe(
            table, hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-row",
            column_config={"latency_ms": "latency", "total_tokens": "tokens",
                           "time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")},
        )
        rows = selection.selection.rows
        if rows:
            self.trace_detail(table.iloc[rows[0]]["trace_id"])
        else:
            st.caption("Select a row to inspect its spans.")

    def trace_detail(self, trace_id: str) -> None:
        trace = self.api_get(f"/observability/traces/{trace_id}")
        st.subheader(f"Trace `{trace_id}`")
        st.markdown(f"**Q:** {trace['message']}")
        st.markdown(f"**A:** {trace['response'] or '_(empty)_'}")
        if trace.get("error"):
            st.error(trace["error"])

        spans = pd.DataFrame(trace["spans"])
        if spans.empty:
            st.caption("No spans recorded for this trace.")
            return

        t0 = spans["start_ts"].min()
        spans["start_s"] = spans["start_ts"] - t0
        spans["end_s"] = spans["start_s"] + spans["latency_ms"] / 1000
        spans["label"] = spans["kind"] + ": " + spans["name"]
        st.altair_chart(
            alt.Chart(spans).mark_bar(cornerRadius=3).encode(
                x=alt.X("start_s:Q", title="seconds since start"),
                x2="end_s:Q",
                y=alt.Y("span_id:N", sort=alt.SortField("start_ts"), title=None,
                        axis=alt.Axis(labels=False, ticks=False)),
                color=alt.Color("kind:N", title="span kind"),
                tooltip=["label", "model", "status", alt.Tooltip("latency_ms:Q", format=".0f"),
                         "input_tokens", "output_tokens"],
            ).properties(height=max(120, 28 * len(spans))),
            width="stretch",
        )

        for _, span in spans.iterrows():
            icon = "✅" if span["status"] == "ok" else "❌"
            tokens = ""
            if pd.notna(span.get("input_tokens")):
                tokens = f" · {int(span['input_tokens'])}→{int(span['output_tokens'] or 0)} tokens"
            with st.expander(f"{icon} {span['label']} · {fmt_ms(span['latency_ms'])}{tokens}"):
                if span.get("model"):
                    st.caption(f"model: {span['model']}")
                if span.get("error"):
                    st.error(span["error"])
                st.markdown("**Input**")
                st.code(span["input"] or "", language=None, wrap_lines=True)
                st.markdown("**Output**")
                st.code(span["output"] or "", language=None, wrap_lines=True)

    # ------------------------------------------------------------ Navigation
    def run(self, chat_page: Callable[[], None], sidebar_extra: Callable[[Optional[dict]], None] = None) -> None:
        info = self.backend_info()
        with st.sidebar:
            self.sidebar_status(info)
            if sidebar_extra:
                sidebar_extra(info)
            self.sidebar_langsmith(info)
            st.divider()
        nav = st.navigation([
            st.Page(chat_page, title="Chat", icon="💬", default=True),
            st.Page(self.observability_page, title="Observability", icon="📈", url_path="observability"),
        ])
        nav.run()


# ---------------------------------------------------------------- small helpers
def fmt_ms(ms) -> str:
    if ms is None or pd.isna(ms):
        return "–"
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


def experts_used(intent: Optional[str]) -> List[str]:
    """ "disease+fertilize" -> ["disease", "fertilize"] (the supervisor variants join with +)."""
    return [i for i in (intent or "unknown").split("+") if i]


def expert_badges(intent: Optional[str], prefix: str = "") -> None:
    badges = []
    for i in experts_used(intent):
        label, color = INTENT_LABELS.get(i, (i, "gray"))
        badges.append(f":{color}-badge[{label}]")
    st.markdown((prefix + " " if prefix else "") + " ".join(badges))
