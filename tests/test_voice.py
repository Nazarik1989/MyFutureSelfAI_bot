from types import SimpleNamespace

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.transcription import DisabledTranscriptionService, TranscriptionError


class FailingTranscription:
    async def transcribe(self, audio: bytes, filename: str) -> str:
        raise TranscriptionError("private provider detail")


class FakeProgress:
    def __init__(self):
        self.text = ""

    async def edit_text(self, text: str):
        self.text = text


class FakeMessage:
    def __init__(self):
        self.progress = FakeProgress()
        self.voice = SimpleNamespace(
            duration=10,
            file_size=3,
            get_file=self.get_file,
        )
        self.audio = None

    async def reply_text(self, text: str):
        self.progress.text = text
        return self.progress

    async def get_file(self):
        return SimpleNamespace(download_as_bytearray=self.download)

    async def download(self):
        return bytearray(b"ogg")


async def test_recognition_error_is_safe_for_user(db, fake_ai):
    settings = Settings(telegram_bot_token="test", ai_api_key="test", ai_model="test-model")
    bot = FutureSelfBot(settings, db, fake_ai, FailingTranscription())
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message)
    await bot.voice(update, SimpleNamespace(user_data={}))
    assert "Не удалось распознать" in message.progress.text
    assert "private provider detail" not in message.progress.text


async def test_disabled_transcription_keeps_bot_running(db, fake_ai):
    settings = Settings(
        telegram_bot_token="test",
        ai_api_key="test",
        ai_model="test-model",
        transcription_provider="disabled",
    )
    bot = FutureSelfBot(settings, db, fake_ai, DisabledTranscriptionService())
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message)
    await bot.voice(update, SimpleNamespace(user_data={}))
    assert "временно не настроено" in message.progress.text


def test_voice_enabled_from_explicit_env_file(tmp_path, db, fake_ai):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "TELEGRAM_BOT_TOKEN=123456:TEST",
                "AI_PROVIDER=openrouter",
                "AI_API_KEY=test-key",
                "AI_MODEL=test-model",
                "TRANSCRIPTION_PROVIDER=openai",
                "TRANSCRIPTION_API_KEY=separate-test-key",
                "TRANSCRIPTION_BASE_URL=https://api.openai.com/v1",
                "TRANSCRIPTION_MODEL=gpt-4o-transcribe",
                "ENABLE_VOICE=true",
            )
        ),
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    bot = FutureSelfBot(settings, db, fake_ai, FailingTranscription())
    assert settings.enable_voice is True
    assert settings.transcription_provider == "openai"
    assert bot.voice_enabled is True
