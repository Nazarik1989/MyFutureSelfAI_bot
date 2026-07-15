from io import BytesIO
from typing import Protocol

from openai import AsyncOpenAI

from .config import Settings


class TranscriptionError(RuntimeError):
    pass


class TranscriptionDisabledError(TranscriptionError):
    pass


class TranscriptionService(Protocol):
    enabled: bool

    async def transcribe(self, audio: bytes, filename: str) -> str: ...


class DisabledTranscriptionService:
    enabled = False

    async def transcribe(self, audio: bytes, filename: str) -> str:
        raise TranscriptionDisabledError("Transcription is disabled")


class OpenAICompatibleTranscriptionService:
    enabled = True

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def transcribe(self, audio: bytes, filename: str) -> str:
        file = BytesIO(audio)
        file.name = filename
        try:
            result = await self.client.audio.transcriptions.create(model=self.model, file=file)
        except Exception as exc:
            raise TranscriptionError("Speech-to-text failed") from exc
        text = result.text.strip()
        if not text:
            raise TranscriptionError("Speech-to-text returned empty text")
        return text


def create_transcription_service(settings: Settings) -> TranscriptionService:
    if settings.transcription_provider == "disabled":
        return DisabledTranscriptionService()
    api_key = settings.transcription_api_key
    if settings.transcription_provider == "local" and not api_key:
        api_key = "local-not-secret"
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings.transcription_base_url,
    )
    return OpenAICompatibleTranscriptionService(client, settings.transcription_model)
