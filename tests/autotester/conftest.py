import socket

import pytest


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail before any autotester scenario can open a real network connection."""

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    loopback_hosts = {"127.0.0.1", "::1", "localhost"}

    def host_from(address: object) -> str | None:
        if isinstance(address, tuple) and address:
            return str(address[0]).lower()
        return None

    def guarded_connect(sock: socket.socket, address: object) -> None:
        if host_from(address) in loopback_hosts:
            return original_connect(sock, address)
        raise AssertionError("External network access is forbidden in tests/autotester")

    def guarded_connect_ex(sock: socket.socket, address: object) -> int:
        if host_from(address) in loopback_hosts:
            return original_connect_ex(sock, address)
        raise AssertionError("External network access is forbidden in tests/autotester")

    def guarded_create_connection(
        address: object, *args: object, **kwargs: object
    ) -> socket.socket:
        if host_from(address) in loopback_hosts:
            return original_create_connection(address, *args, **kwargs)
        raise AssertionError("External network access is forbidden in tests/autotester")

    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
