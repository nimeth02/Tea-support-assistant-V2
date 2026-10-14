"""Streamlit frontend for the Tea Support Assistant (chat + observability dashboard)."""
import os
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("TEA_API_URL", "http://127.0.0.1:8001")
CHAT_TIMEOUT = 300  # agents are rate limited, so a single chat can take a while

EXAMPLES = [
    "When was my last fertilization?",
    "How much fertilizer should I use?",
    "Will it rain on my tea land this week?",
    "My tea leaves have brown spots, what should I do?",
    "Which dates does the company not collect tea leaves?",
]

st.set_page_config(page_title="Tea Support Assistant", page_icon="🍃", layout="wide")


# ---------------------------------------------------------------- API helpers
def api_get(path: str, **params):
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def backend_online() -> bool:
    try:
        requests.get(f"{API_URL}/welcome", timeout=3).raise_for_status()
        return True
    except requests.RequestException:
        return False


def send_chat(message: str, image_file=None) -> dict:
    files = {"image": (image_file.name, image_file.getvalue(), image_file.type)} if image_file else None
    resp = requests.post(f"{API_URL}/chat", data={"message": message}, files=files, timeout=CHAT_TIMEOUT)
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get(
            "content-type", "").startswith("application/json") else resp.text
        return {"response": f"⚠️ Backend error ({resp.status_code}): {detail}",
                "intent": "error", "trace_id": resp.headers.get("X-Trace-Id")}
    return resp.json()


def fmt_ms(ms) -> str:
    if ms is None or pd.isna(ms):
        return "–"
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"


# ---------------------------------------------------------------- Chat page
def chat_page():
    st.title("🍃 Tea Support Assistant")
    st.caption("Ask about fertilizer, weather for your land, tea diseases (or upload a leaf photo), "
               "or company policies.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    with st.sidebar:
        st.subheader("Try asking")
        for example in EXAMPLES:
            if st.button(example, width="stretch"):
                st.session_state.pending = example
        st.divider()
        if st.button("🗑️ Clear chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    # ChatGPT-style attachment: a small paperclip that opens the uploader, instead of a
    # permanently-visible drop zone. The attached file shows as a compact chip, not the full
    # uploader box, once chosen.
    attach_col, chip_col = st.columns([0.07, 0.93])
    with attach_col:
        with st.popover("📎", help="Attach a tea leaf photo"):
            uploaded_image = st.file_uploader(
                "Tea leaf photo for disease identification",
                type=["jpg", "jpeg", "png"],
                key=f"uploader_{st.session_state.uploader_key}",
            )
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

    prompt = st.chat_input("Type your question…") or st.session_state.pop("pending", None)
    if not prompt and not uploaded_image:
        return
    prompt = prompt or "What disease or pest is affecting this tea leaf?"

    if uploaded_image:
        st.session_state.uploader_key += 1  # clear the attachment now that it's being sent

    display_text = prompt + (" 📷" if uploaded_image else "")
    st.session_state.messages.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking… (free-tier Gemini is rate limited, this can take ~30s)"):
            try:
                data = send_chat(prompt, uploaded_image)
            except requests.RequestException as e:
                data = {"response": f"⚠️ Could not reach the backend at {API_URL}: {e}",
                        "intent": "error"}
        answer = data.get("response") or "_No answer — the model may be rate limited, try again in a minute._"
        meta = " · ".join(filter(None, [
            f"intent: {data.get('intent')}",
            f"latency: {fmt_ms(data['latency_ms'])}" if data.get("latency_ms") else None,
            f"trace: {data['trace_id']}" if data.get("trace_id") else None,
        ]))
        st.markdown(answer)
        st.caption(meta)
    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})


# ---------------------------------------------------------------- Observability page
def observability_page():
    head, refresh = st.columns([6, 1])
    head.title("📈 Observability")
    refresh.button("🔄 Refresh", width="stretch")

    try:
        stats = api_get("/observability/stats")
        traces = pd.DataFrame(api_get("/observability/traces", limit=500))
        spans = pd.DataFrame(api_get("/observability/spans", limit=5000))
    except requests.RequestException as e:
        st.error(f"Could not load observability data from {API_URL}: {e}")
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
            alt.Chart(intents).mark_bar().encode(
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
        trace_detail(table.iloc[rows[0]]["trace_id"])
    else:
        st.caption("Select a row to inspect its spans.")


def trace_detail(trace_id: str):
    trace = api_get(f"/observability/traces/{trace_id}")
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


# ---------------------------------------------------------------- Navigation
with st.sidebar:
    if backend_online():
        st.success(f"Backend online · {API_URL}", icon="🟢")
    else:
        st.error(f"Backend offline · {API_URL}", icon="🔴")

nav = st.navigation([
    st.Page(chat_page, title="Chat", icon="💬", default=True),
    st.Page(observability_page, title="Observability", icon="📈", url_path="observability"),
])
nav.run()
