"""Shared OpenAI client; the Langfuse wrapper traces every call made through it."""
from langfuse.openai import OpenAI

_client = None


def client() -> OpenAI:
    global _client
    _client = _client or OpenAI()
    return _client
