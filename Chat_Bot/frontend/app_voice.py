"""UI for the multilingual + voice variant (Chat_Bot/agent multilangual and voice also).

How this variant works: the same LangGraph router as the base version, with a translation step
on each side. The farmer asks in Sinhala, Tamil or English - typed or spoken. The question is
turned into English for the experts (whose prompts and knowledge bases are English), and the
answer is translated back and, if wanted, read out loud.

Run:  python launch.py up voice      (UI on http://localhost:8515)
"""
import requests
import streamlit as st

from ui_common import UI, INTENT_LABELS

ui = UI("voice")
ui.page_setup("🗣️")

EXAMPLES = [
    "මගේ තේ කොළවල දුඹුරු පැල්ලම් තියෙනවා. මොකද කරන්න ඕනේ?",      # Sinhala: brown spots on my leaves
    "நான் கடைசியாக எப்போது உரம் இட்டேன்?",                          # Tamil: when did I last fertilize?
    "How much fertilizer should I use?",
    "සමාගම තේ දළු එකතු නොකරන දින මොනවාද?",                          # Sinhala: which days no leaf collection?
]

# Used when the backend is offline and we can't ask it for its language list.
DEFAULT_LANGUAGES = [
    {"code": "en", "name": "English", "native": "English"},
    {"code": "si", "name": "Sinhala", "native": "සිංහල"},
    {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
]
LANG_NATIVE = {l["code"]: l["native"] for l in DEFAULT_LANGUAGES}

# intent -> the expert node it routes to (mirrors route_by_intent in orchestrator_agent.py)
ROUTES = {"disease": "disease_agent", "fertilize": "fertilize_agent",
          "company_help": "help_agent", "help": "help_agent"}


def path_of(data: dict) -> list:
    """The nodes the last question visited, in order."""
    expert = "disease_agent" if data.get("has_image") else ROUTES.get(data.get("intent"))
    return ["detect_language", "classify_intent"] + ([expert] if expert else []) + ["translate_response"]


def graph_dot(path: list = None) -> str:
    """The LangGraph workflow as a Graphviz diagram; the path of the last question is highlighted."""
    color = ui.variant.color
    path = path or []

    def node(name, label):
        on = name in path
        style = f'style="filled", fillcolor="{color}", fontcolor="white"' if on else 'style="filled", fillcolor="#f1f3f1"'
        return f'{name} [label="{label}", {style}];'

    def edge(a, b, label=""):
        on = a in path and b in path and path.index(b) == path.index(a) + 1
        attrs = f'color="{color}", penwidth=3' if on else 'color="#9e9e9e"'
        return f'{a} -> {b} [label="{label}", {attrs}];'

    return f"""digraph G {{
        rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=9];
        START [shape=circle, label="start", fillcolor="#e0e0e0"];
        {node("detect_language", "detect_language\\n🌐 si / ta / en → English")}
        {node("classify_intent", "classify_intent\\n(LLM picks 1 intent)")}
        {node("disease_agent", "disease_agent\\n🦠 RAG + web + photo")}
        {node("fertilize_agent", "fertilize_agent\\n🌱 dosage, weather, history")}
        {node("help_agent", "help_agent\\n🏢 company RAG")}
        {node("translate_response", "translate_response\\n🌐 English → si / ta")}
        END [shape=circle, label="end", fillcolor="#e0e0e0"];
        START -> detect_language [color="#9e9e9e"];
        {edge("detect_language", "classify_intent")}
        {edge("classify_intent", "disease_agent", "disease / photo")}
        {edge("classify_intent", "fertilize_agent", "fertilize")}
        {edge("classify_intent", "help_agent", "company_help")}
        {edge("classify_intent", "translate_response", "unknown")}
        {edge("disease_agent", "translate_response")}
        {edge("fertilize_agent", "translate_response")}
        {edge("help_agent", "translate_response")}
        translate_response -> END [color="{color if path else '#9e9e9e'}"{", penwidth=3" if path else ""}];
    }}"""


def intro():
    ui.banner("🗣️", "Ask in Sinhala, Tamil or English · type it or say it · hear the answer read back")
    last = st.session_state.get("last_reply")
    with st.expander("How this version works", expanded=not st.session_state.get("messages")):
        st.markdown(
            "The experts (disease, fertilizer, company help) only understand **English**: their "
            "instructions and knowledge bases are written in English. So this version wraps them in "
            "two translation steps:\n\n"
            "1. **detect_language** works out which language you used (Sinhala and Tamil have their "
            "own alphabets, so this needs no AI call) and translates your question into English.\n"
            "2. The usual **classify_intent → expert** route answers it in English.\n"
            "3. **translate_response** translates the answer back into your language.\n\n"
            "**Voice** happens outside the graph, in the backend: a recording is first turned into "
            "text (*speech-to-text*), and the final answer can be turned into an MP3 (*text-to-speech*) "
            "that plays under the reply.\n\n"
            "- ✅ English questions skip both translation steps, so they cost nothing extra.\n"
            "- ⚠️ Sinhala/Tamil questions need two extra AI calls, so they are slower on the free tier.\n"
            "- ⚠️ Romanized Sinhala/Tamil (Latin letters) looks like English to the detector - pick "
            "your language in the sidebar instead.\n\n"
            "The highlighted path is the route your **last** question took."
        )
        st.graphviz_chart(graph_dot(path_of(last) if last else None))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_audio(url: str):
    """The MP3 of a spoken answer, cached so re-draws don't download it again."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None


def details(data: dict):
    intent = data.get("intent") or "unknown"
    label, color = INTENT_LABELS.get(intent, (intent, "gray"))
    lang = data.get("language") or "en"
    badges = [f":{color}-badge[{label}]", f":violet-badge[🌐 {LANG_NATIVE.get(lang, lang)}]"]
    if data.get("transcript"):
        badges.append(":orange-badge[🎤 voice question]")
    if data.get("audio_url"):
        badges.append(":blue-badge[🔊 spoken answer]")
    st.markdown(" ".join(badges))

    if data.get("transcript") or data.get("translated_query"):
        with st.expander("What happened to your question"):
            if data.get("transcript"):
                st.markdown(f"**🎤 Heard:** {data['transcript']}")
            if data.get("translated_query"):
                st.markdown(f"**🌐 Translated for the experts:** {data['translated_query']}")
                st.caption("The experts answered this English version; the reply above was translated back.")

    if data.get("audio_url"):
        audio = fetch_audio(ui.api_url + data["audio_url"])
        if audio:
            # Play the newest answer once, automatically; older answers get a play button only.
            is_newest = data is st.session_state.get("last_reply")
            autoplay = is_newest and st.session_state.get("autoplayed") != data.get("trace_id")
            if autoplay:
                st.session_state.autoplayed = data.get("trace_id")
            st.audio(audio, format="audio/mp3", autoplay=autoplay)
        else:
            st.caption("🔇 The spoken answer could not be loaded from the backend.")


def composer(key: str):
    """The microphone next to the paperclip. Stopping the recording sends it straight away."""
    return st.audio_input(
        "Record your question", key=key, label_visibility="collapsed",
        help="Click the microphone, ask your question in Sinhala, Tamil or English, then click stop. "
             "The recording is sent as soon as you stop. Attach a leaf photo first if you want to send both.",
    )


def request_fields() -> dict:
    return {"language": st.session_state.get("language", "auto"),
            "speak": st.session_state.get("speak", True)}


def sidebar(info):
    st.subheader("🌐 Language & voice")
    features = (info or {}).get("features") or {}
    langs = features.get("languages") or DEFAULT_LANGUAGES
    labels = {"auto": "Auto-detect from the message"}
    for l in langs:
        labels[l["code"]] = l["name"] if l["native"] == l["name"] else f"{l['native']} · {l['name']}"
    st.radio("Answer in", ["auto"] + [l["code"] for l in langs], key="language", format_func=labels.get,
             help="Auto-detect recognises Sinhala and Tamil script. If you type Sinhala or Tamil with "
                  "Latin letters, choose the language here so it is translated correctly.")
    st.toggle("🔊 Read answers aloud", key="speak", value=True,
              help="The backend turns the answer into speech (MP3) and it plays under the reply.")

    speech = (info or {}).get("speech")
    if speech:
        st.caption(f"Speech in: `{speech['stt_provider']}` · Speech out: `{speech['tts_provider']}` "
                   f"(STT_PROVIDER / TTS_PROVIDER in `.env`)")
    elif info:
        st.warning("This backend has no voice support - start the voice variant with "
                   "`python launch.py up voice`.")

    last = st.session_state.get("last_reply")
    if last:
        st.caption(f"Last answer in **{LANG_NATIVE.get(last.get('language'), last.get('language'))}**"
                   + (" · from a voice question" if last.get("transcript") else ""))


ui.run(lambda: ui.chat_page(EXAMPLES, intro, details,
                            "Listening, translating and routing… (free-tier Gemini is rate limited, "
                            "Sinhala/Tamil questions can take ~45s)",
                            composer=composer, request_fields=request_fields),
       sidebar)
