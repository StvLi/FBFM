import pytest

import dreamzero_fbfm.server as server_module


class FakeConnection:
    def __init__(self):
        self.messages = [{"type": "close"}]
        self.responses = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def settimeout(self, _timeout):
        pass

    def close(self):
        self.closed = True


class FakeSocket:
    def __init__(self, connections):
        self.connections = list(connections)
        self.accept_count = 0
        self.closed = False

    def setsockopt(self, *_):
        pass

    def bind(self, _address):
        pass

    def listen(self, _backlog):
        pass

    def accept(self):
        self.accept_count += 1
        if self.connections:
            return self.connections.pop(0), ("127.0.0.1", 10000 + self.accept_count)
        raise KeyboardInterrupt

    def close(self):
        self.closed = True


class FakeAudit:
    def __init__(self):
        self.events = []

    def write(self, event, **values):
        self.events.append((event, values))


class FakeRuntime:
    def __init__(self):
        self.cancel_count = 0

    def cancel(self):
        self.cancel_count += 1


def test_server_accepts_multiple_sequential_clients(monkeypatch):
    connections = [FakeConnection(), FakeConnection()]
    listening_socket = FakeSocket(connections)
    monkeypatch.setattr(server_module.socket, "socket", lambda *_args, **_kwargs: listening_socket)
    monkeypatch.setattr(
        server_module,
        "receive_message",
        lambda connection: connection.messages.pop(0) if connection.messages else None,
    )
    monkeypatch.setattr(
        server_module,
        "send_message",
        lambda connection, response: connection.responses.append(response),
    )
    monkeypatch.setattr(
        server_module.ModelServer,
        "_handle",
        lambda _self, request_type, _request: {"status": "ok", "type": request_type},
    )

    server = object.__new__(server_module.ModelServer)
    server.host = "127.0.0.1"
    server.port = 18766
    server.audit = FakeAudit()
    server.runtime = FakeRuntime()
    server.job = None

    with pytest.raises(KeyboardInterrupt):
        server.serve()

    assert listening_socket.accept_count == 3
    assert listening_socket.closed
    assert all(connection.closed for connection in connections)
    assert all(connection.responses == [{"status": "ok", "type": "close"}] for connection in connections)
    assert [event for event, _ in server.audit.events].count("client_connected") == 2
    assert [event for event, _ in server.audit.events].count("client_disconnected") == 2
