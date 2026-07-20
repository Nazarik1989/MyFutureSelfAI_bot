from itertools import count
from typing import Any

from future_self.schemas import AssistantAnswer, IntentResult


class UnexpectedLLMCall(AssertionError):
    pass


class StrictAI:
    """Deterministic AI double that rejects every unstubbed prompt."""

    def __init__(self, responses: dict[str, IntentResult]):
        self.responses = responses
        self.route_calls: list[str] = []

    async def route_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        conversation_context: dict[str, object] | None = None,
    ) -> IntentResult:
        self.route_calls.append(text)
        try:
            response = self.responses[text]
        except KeyError as exc:
            raise UnexpectedLLMCall(f"Unexpected LLM route for {text!r}") from exc
        return response.model_copy(deep=True)

    async def answer_message(
        self,
        text: str,
        temporal_context: dict[str, str],
        conversation_context: dict[str, object] | None = None,
    ) -> AssistantAnswer:
        raise UnexpectedLLMCall(f"Unexpected LLM answer for {text!r}")


class ScriptedTranscription:
    enabled = True

    def __init__(self) -> None:
        self._next_text: str | None = None
        self.calls: list[tuple[int, str]] = []

    def queue(self, text: str) -> None:
        if self._next_text is not None:
            raise AssertionError("Previous transcription was not consumed")
        self._next_text = text

    async def transcribe(self, audio: bytes, filename: str) -> str:
        if self._next_text is None:
            raise AssertionError("Voice step did not provide a transcription")
        text, self._next_text = self._next_text, None
        self.calls.append((len(audio), filename))
        return text


class FakeFile:
    async def download_as_bytearray(self) -> bytearray:
        return bytearray(b"autotest-voice")


class FakeVoice:
    duration = 2
    file_size = 14
    mime_type = "audio/ogg"
    file_name = "autotest.ogg"

    async def get_file(self) -> FakeFile:
        return FakeFile()


class FakeMessage:
    _ids = count(10_000)

    def __init__(self, text: str | None = None, *, voice: FakeVoice | None = None):
        self.text = text
        self.voice = voice
        self.audio = None
        self.reply_to_message = None
        self.replies: list[dict[str, Any]] = []
        self.edits: list[str] = []
        self.message_id = next(self._ids)

    async def reply_text(self, text: str, **kwargs: Any) -> "FakeMessage":
        self.replies.append({"text": text, **kwargs})
        return self

    async def reply_photo(
        self, photo: Any, caption: str | None = None, **kwargs: Any
    ) -> "FakeMessage":
        self.replies.append(
            {
                "text": caption or "",
                "kind": "photo",
                "filename": getattr(photo, "name", None),
                "data": photo.getvalue(),
                **kwargs,
            }
        )
        return self

    async def reply_document(
        self,
        document: Any,
        filename: str | None = None,
        caption: str | None = None,
        **kwargs: Any,
    ) -> "FakeMessage":
        self.replies.append(
            {
                "text": caption or "",
                "kind": "document",
                "filename": filename or getattr(document, "name", None),
                "data": document.getvalue(),
                **kwargs,
            }
        )
        return self

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)


class FakeCallbackQuery:
    def __init__(self, data: str, message: FakeMessage):
        self.data = data
        self.message = message
        self.answers: list[tuple[str | None, bool]] = []
        self.edits: list[str] = []
        self.markup_removed = 0

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str) -> None:
        self.edits.append(text)

    async def edit_message_reply_markup(self, reply_markup: object = None) -> None:
        self.markup_removed += 1


class FakeBot:
    def __init__(self) -> None:
        self.removed: list[tuple[int, int]] = []

    async def edit_message_reply_markup(
        self,
        *,
        chat_id: int,
        message_id: int,
        reply_markup: object,
    ) -> None:
        self.removed.append((chat_id, message_id))
