import base64
import os
import tempfile
from typing import Optional

import httpx

# Optional imports guarded for environments without these packages
try:
    from deep_translator import GoogleTranslator  # type: ignore
except Exception:  # pragma: no cover
    GoogleTranslator = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def translate_text(text: str, target_lang: str = "en", source_lang: Optional[str] = None) -> str:
    """
    Translate text using deep-translator's GoogleTranslator when available.
    Falls back to returning the original text if translator is unavailable.
    """
    if not text:
        return text
    try:
        if GoogleTranslator is not None:
            if source_lang:
                return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            return GoogleTranslator(target=target_lang).translate(text)
    except Exception:
        pass
    return text


def asr_transcribe(audio_b64: str, language: Optional[str] = None) -> str:
    """
    Transcribe audio using OpenAI Whisper API when OPENAI_API_KEY is set.
    Input is base64-encoded audio (wav/m4a/mp3). Falls back to a stub.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return "ASR unavailable in this environment."

    client = OpenAI(api_key=api_key)

    # Decode base64 into a temporary file
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return "Invalid audio encoding"

    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        try:
            # Whisper-1 transcription
            with open(tmp.name, "rb") as f:
                result = client.audio.transcriptions.create(model="whisper-1", file=f, language=language)
            # openai>=1.0 returns .text on object
            text = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else None)
            return text or ""
        except Exception:
            return ""


async def fetch_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
    except Exception:
        return None
    return None
