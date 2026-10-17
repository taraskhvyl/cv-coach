"""Speech-to-text and text-to-speech."""
import tempfile


from langfuse import observe

from . import config
from .llm import client


@observe(name="transcribe")
def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        return client().audio.transcriptions.create(model=config.STT_MODEL, file=f).text


@observe(name="speak")
def speak(text: str) -> str:
    audio = client().audio.speech.create(model=config.TTS_MODEL, voice=config.TTS_VOICE, input=text[:4000])
    path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    with open(path, "wb") as f:
        f.write(audio.content)
    return path
