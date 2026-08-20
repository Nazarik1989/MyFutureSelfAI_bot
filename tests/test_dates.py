from datetime import UTC, datetime

from future_self.dates import DateResolver


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
