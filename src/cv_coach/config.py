import os

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
EMBED_MODEL = "text-embedding-3-small"
STT_MODEL = "whisper-1"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
