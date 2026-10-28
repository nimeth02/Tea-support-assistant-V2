"""Prompts for the two translation steps that wrap the expert agents, and for transcription.

Why translate at all: the intent classifier, the expert prompts and the RAG knowledge bases
(and their embeddings) are all in English. So the graph translates a Sinhala/Tamil question
*into* English first, lets the English-only experts work, then translates the answer *back*.

  translate_to_english_prompt    farmer's message (any supported language) -> English
  translate_from_english_prompt  expert's English answer -> farmer's language
  transcribe_prompt              text instructions sent with an audio clip to the Gemini model
                                 when STT_PROVIDER=gemini (speech-to-text)

Each expects `language` = the full language name (e.g. "Sinhala"), and `text`.
"""
from langchain_core.prompts import ChatPromptTemplate

TRANSLATE_TO_ENGLISH_TEMPLATE = """<role>
You are a translator for a support assistant used by small-scale tea farmers in Sri Lanka.
</role>

<instructions>
Translate the farmer's message inside <text> from {language} into clear, natural English.
The message may be written in {language} script or in romanized {language} (Latin letters).
If part of it is already English, keep that part as it is.
</instructions>

<constraints>
- Keep numbers, units, dates, place names and product names (e.g. fertilizer names) unchanged.
- Do not answer the question, add explanations, or leave anything out.
- Reply with the English translation only - no quotes, no labels, no notes.
</constraints>

<text>
{text}
</text>"""

TRANSLATE_FROM_ENGLISH_TEMPLATE = """<role>
You are a translator for a support assistant used by small-scale tea farmers in Sri Lanka.
</role>

<instructions>
Translate the assistant's reply inside <text> from English into {language}.
Use simple, everyday {language} that a farmer would use in conversation, not formal or literary language.
</instructions>

<constraints>
- Keep numbers, units, dates, prices and product names (e.g. fertilizer names like urea, T750) unchanged.
- Keep the structure: the same line breaks, bullet points and markdown markers (like ** or -).
- Do not add, remove or soften any advice.
- Reply with the {language} translation only - no quotes, no labels, no notes.
</constraints>

<text>
{text}
</text>"""

# Speech-to-text instructions for the Gemini model (the audio clip is attached as a second
# content part). We ask for JSON so the language it heard can be read back reliably.
TRANSCRIBE_TEMPLATE = """<role>
You transcribe voice messages from small-scale tea farmers in Sri Lanka. The farmer may speak
Sinhala, Tamil or English (sometimes mixed).
</role>

<instructions>
1. Listen to the attached audio and write down exactly what was said, in the script of the
   language spoken (Sinhala script for Sinhala, Tamil script for Tamil).
2. Identify the main language: "si" (Sinhala), "ta" (Tamil), "en" (English) or "other".
{language_hint}
</instructions>

<constraints>
- Do not answer, summarise or correct the farmer. Transcribe only.
- If nothing understandable is said, use an empty string for "text".
</constraints>

<output_format>
Reply with one JSON object and nothing else, like: {{"language": "si", "text": "..."}}
</output_format>"""

translate_to_english_prompt = ChatPromptTemplate.from_template(TRANSLATE_TO_ENGLISH_TEMPLATE)
translate_from_english_prompt = ChatPromptTemplate.from_template(TRANSLATE_FROM_ENGLISH_TEMPLATE)
transcribe_prompt = ChatPromptTemplate.from_template(TRANSCRIBE_TEMPLATE)
