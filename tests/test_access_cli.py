import asyncio
from io import StringIO

import pytest

from future_self.access_cli import AccessCLISettings, run_cli
from future_self.db import Database
from future_self.repositories import UserRepository


def make_database(tmp_path, *, display_name: str | None = None) -> str:
    path = tmp_path / "access-cli.db"
    url = f"sqlite+aiosqlite:///{path.as_posix()}"

    async def prepare() -> None:
        db = Database(url)
        await db.create_all_for_tests()
        async with db.session() as session:
            user = await UserRepository(session).get_or_create(530129470, "Europe/Moscow")
            user.display_name = display_name
        await db.dispose()

    asyncio.run(prepare())
    return url


def invoke(url: str, *args: str) -> tuple[int, str, str]:
    output = StringIO()
    error = StringIO()
    code = run_cli(
        args,
        settings=AccessCLISettings(_env_file=None, database_url=url),
        output=output,
        error=error,
    )
    return code, output.getvalue(), error.getvalue()


def test_cli_status_and_all_mutations(tmp_path):
    url = make_database(tmp_path)

    code, output, error = invoke(url, "status", "530129470")
    assert code == 0 and not error
    assert "access_tier=guest" in output
    assert "access_version=1" in output
    assert "result=no-op" in output

    expected = [
        (("grant", "subscriber", "530129470"), "subscriber", 2),
        (("grant", "admin", "530129470"), "admin", 3),
        (("set", "guest", "530129470"), "guest", 4),
        (("block", "530129470"), "blocked", 5),
        (("unblock", "530129470"), "guest", 6),
    ]
    for args, tier, version in expected:
        code, output, error = invoke(url, *args)
        assert code == 0 and not error
        assert f"access_tier={tier}" in output
        assert f"access_version={version}" in output
        assert "result=changed" in output

    code, output, error = invoke(url, "unblock", "530129470")
    assert code == 0 and not error
    assert "access_version=6" in output
    assert "result=no-op" in output


def test_cli_not_found_does_not_create_user(tmp_path):
    url = make_database(tmp_path)
    code, output, error = invoke(url, "grant", "subscriber", "999999")
    assert code != 0
    assert not output
    assert error == "access user not found: telegram_id=999999\n"
    code, output, error = invoke(url, "status", "999999")
    assert code != 0 and not output
    assert error == "access user not found: telegram_id=999999\n"


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_cli_rejects_invalid_telegram_id(value):
    with pytest.raises(SystemExit) as stopped:
        run_cli(["status", value], settings=AccessCLISettings(_env_file=None))
    assert stopped.value.code != 0


def test_cli_output_omits_profile_and_configuration_secrets(tmp_path):
    private_name = "PRIVATE-DISPLAY-NAME"
    url = make_database(tmp_path, display_name=private_name)
    code, output, error = invoke(url, "status", "530129470")
    combined = output + error
    assert code == 0
    assert private_name not in combined
    assert url not in combined
    assert "token" not in combined.casefold()
    assert set(line.split("=", 1)[0] for line in output.splitlines()) == {
        "telegram_id",
        "access_tier",
        "access_version",
        "onboarding_completed",
        "result",
    }


def test_minimal_cli_settings_require_no_bot_or_provider_secrets(monkeypatch, tmp_path):
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "AI_API_KEY",
        "OPENAI_API_KEY",
        "TRANSCRIPTION_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    database = tmp_path / "minimal.db"
    settings = AccessCLISettings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{database.as_posix()}",
    )
    assert settings.database_url.endswith("minimal.db")
