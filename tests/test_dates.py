from datetime import UTC, datetime, time

import pytest

from future_self.dates import DateResolver


@pytest.mark.parametrize(
    "text",
    [
        "в 22ч пора спать",
        "в 22 ч пора спать",
        "пора спать в 22:00",
        "пора спать в 22.00",
        "в десять вечера пора спать",
        "пора спать в десять вечера",
    ],
)
def test_extract_local_time_accepts_grounded_conversational_clock_forms(text) -> None:
    assert DateResolver.extract_local_time(text) == time(22)


@pytest.mark.parametrize("text", ["05.09.2026", "5.9.2026", "2026-09-05"])
def test_extract_local_time_never_treats_a_full_date_span_as_a_clock(text) -> None:
    assert DateResolver.extract_local_time(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "05.09.2026 в 22:00",
        "в 22.00 на 5.9.2026",
        "2026-09-05, затем в 22:00",
    ],
)
def test_extract_local_time_uses_the_clock_outside_a_full_date_span(text) -> None:
    assert DateResolver.extract_local_time(text) == time(22)


def test_resolve_uses_injected_clock_when_now_is_omitted() -> None:
    fixed_now = datetime(2035, 12, 31, 21, 30, tzinfo=UTC)
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        calls += 1
        return fixed_now

    result = DateResolver(now_provider=clock).resolve("завтра", "Europe/Moscow")

    assert result.status == "resolved"
    assert result.target_date is not None
    assert result.target_date.isoformat() == "2036-01-02"
    assert calls == 1


def test_resolve_supplied_now_takes_precedence_over_injected_clock() -> None:
    def unexpected_clock() -> datetime:
        raise AssertionError("injected clock must not run when now is supplied")

    result = DateResolver(now_provider=unexpected_clock).resolve(
        "завтра",
        "Europe/Moscow",
        now=datetime(2040, 5, 10, 21, 30, tzinfo=UTC),
    )

    assert result.status == "resolved"
    assert result.target_date is not None
    assert result.target_date.isoformat() == "2040-05-12"
