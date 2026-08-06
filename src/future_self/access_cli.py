from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import TextIO

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .access import AccessMutationResult, AccessService, AccessStatus, AccessUserNotFound
from .config import ENV_FILE_PATH
from .db import Database


class AccessCLISettings(BaseSettings):
    """Minimal operator settings with no Telegram, AI, or STT requirements."""

    database_url: str = "sqlite+aiosqlite:///./future_self.db"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("database_url")
    @classmethod
    def async_database_driver(cls, value: str) -> str:
        clean = value.strip()
        if clean.startswith("postgresql://"):
            return clean.replace("postgresql://", "postgresql+asyncpg://", 1)
        return clean


def positive_telegram_id(value: str) -> int:
    try:
        telegram_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("telegram_id must be a positive integer") from exc
    if telegram_id <= 0:
        raise argparse.ArgumentTypeError("telegram_id must be a positive integer")
    return telegram_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Future Self access tiers")
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Show an existing user's access status")
    status.add_argument("telegram_id", type=positive_telegram_id)

    grant = commands.add_parser("grant", help="Grant full access")
    grant.add_argument("tier", choices=("subscriber", "admin"))
    grant.add_argument("telegram_id", type=positive_telegram_id)

    set_command = commands.add_parser("set", help="Set a non-full access tier")
    set_command.add_argument("tier", choices=("guest",))
    set_command.add_argument("telegram_id", type=positive_telegram_id)

    block = commands.add_parser("block", help="Block an existing user")
    block.add_argument("telegram_id", type=positive_telegram_id)

    unblock = commands.add_parser("unblock", help="Move a blocked user to guest")
    unblock.add_argument("telegram_id", type=positive_telegram_id)
    return parser


def _print_status(
    status: AccessStatus,
    *,
    changed: bool,
    output: TextIO,
) -> None:
    print(f"telegram_id={status.telegram_id}", file=output)
    print(f"access_tier={status.access_tier}", file=output)
    print(f"access_version={status.access_version}", file=output)
    print(f"onboarding_completed={str(status.onboarding_completed).lower()}", file=output)
    print(f"result={'changed' if changed else 'no-op'}", file=output)


async def _execute(
    args: argparse.Namespace,
    settings: AccessCLISettings,
    *,
    output: TextIO,
    error: TextIO,
) -> int:
    db = Database(settings.database_url)
    service = AccessService(db)
    try:
        if args.command == "status":
            status = await service.status(args.telegram_id)
            if status is None:
                print(f"access user not found: telegram_id={args.telegram_id}", file=error)
                return 4
            _print_status(status, changed=False, output=output)
            return 0

        result: AccessMutationResult
        if args.command == "grant" and args.tier == "subscriber":
            result = await service.grant_subscriber(
                args.telegram_id, source="operator-cli:grant-subscriber"
            )
        elif args.command == "grant" and args.tier == "admin":
            result = await service.grant_admin(args.telegram_id, source="operator-cli:grant-admin")
        elif args.command == "set" and args.tier == "guest":
            result = await service.set_guest(args.telegram_id, source="operator-cli:set-guest")
        elif args.command == "block":
            result = await service.block(args.telegram_id, source="operator-cli:block")
        elif args.command == "unblock":
            result = await service.unblock(args.telegram_id, source="operator-cli:unblock")
        else:  # argparse prevents this branch; keep execution fail-closed.
            print("invalid access command", file=error)
            return 2
        _print_status(result.status, changed=result.changed, output=output)
        return 0
    except AccessUserNotFound:
        print(f"access user not found: telegram_id={args.telegram_id}", file=error)
        return 4
    except Exception:
        # Database exceptions can embed credentials in their messages. Never
        # include exception text or the configured URL in operator output.
        print("access operation failed", file=error)
        return 3
    finally:
        await db.dispose()


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    settings: AccessCLISettings | None = None,
    output: TextIO | None = None,
    error: TextIO | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    stdout = output or sys.stdout
    stderr = error or sys.stderr
    if settings is None:
        try:
            settings = AccessCLISettings()
        except Exception:
            print("access CLI configuration is invalid", file=stderr)
            return 2
    return asyncio.run(_execute(args, settings, output=stdout, error=stderr))


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
