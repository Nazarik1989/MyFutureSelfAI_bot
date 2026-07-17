import socket
from pathlib import Path

import pytest

from .harness import (
    AUTOTEST_TELEGRAM_TOKEN,
    BotAutotester,
    UnsafeAutotestConfiguration,
    assert_safe_runtime,
    build_autotest_settings,
)
from .scenarios import SCENARIOS

pytestmark = pytest.mark.autotester


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
async def test_bot_scenario(tmp_path: Path, scenario) -> None:
    harness = await BotAutotester.create(tmp_path, scenario.llm_stubs)
    try:
        await harness.run(scenario)
    finally:
        await harness.close()


def test_safety_guard_rejects_database_outside_pytest_sandbox(tmp_path: Path) -> None:
    database_url = "sqlite+aiosqlite:////data/future_self.db"
    settings = build_autotest_settings(database_url)

    with pytest.raises(UnsafeAutotestConfiguration, match="inside pytest tmp_path"):
        assert_safe_runtime(settings, database_url, tmp_path)


def test_safety_guard_rejects_non_sentinel_credentials(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}"
    settings = build_autotest_settings(database_url).model_copy(
        update={"telegram_bot_token": f"{AUTOTEST_TELEGRAM_TOKEN}-changed"}
    )

    with pytest.raises(UnsafeAutotestConfiguration, match="sentinel token"):
        assert_safe_runtime(settings, database_url, tmp_path)


def test_external_network_guard_blocks_before_connect() -> None:
    connection = socket.socket()
    try:
        with pytest.raises(AssertionError, match="External network access is forbidden"):
            connection.connect(("203.0.113.1", 443))
    finally:
        connection.close()
