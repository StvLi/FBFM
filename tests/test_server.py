from types import SimpleNamespace

import numpy as np
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


class FakeInferenceRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.chunks = []
        self.feedback = []

    def begin_chunk(self, committed, **values):
        self.chunks.append((committed, values))

    def submit_feedback(self, feedback):
        self.feedback.append(feedback)


class FakePolicy:
    def __init__(self):
        self.action_head = SimpleNamespace(current_start_frame=0)
        self.model_inputs = []

    def predict_action_batch(self, observation, mode):
        assert mode == "eval"
        self.model_inputs.append(observation)
        self.action_head.current_start_frame += 4
        return np.zeros((1, 16, 7), dtype=np.float32), None


def _encoded_observation(value):
    return {
        "main_image": server_module.encode_array(
            np.full((256, 256, 3), value, dtype=np.uint8)
        ),
        "wrist_image": server_module.encode_array(
            np.full((256, 256, 3), value + 10, dtype=np.uint8)
        ),
        "state": server_module.encode_array(np.zeros(8, dtype=np.float32)),
    }


def _inference_server():
    server = object.__new__(server_module.ModelServer)
    server.policy = FakePolicy()
    server.reset_policy_state = lambda policy, seed: setattr(
        policy.action_head, "current_start_frame", 0
    )
    server.audit = FakeAudit()
    server.runtime = FakeInferenceRuntime()
    server.task_description = None
    server.job = None
    server.observation_history = server_module.CausalObservationHistory()
    return server


def test_model_inputs_follow_native_causal_history_and_exclude_feedback():
    server = _inference_server()
    server._handle(
        "reset", {"type": "reset", "task_description": "pick object", "seed": 0}
    )

    server._handle("predict_sync", {"type": "predict_sync", **_encoded_observation(1)})
    server._handle(
        "predict_start",
        {
            "type": "predict_start",
            **_encoded_observation(2),
            "committed_actions": server_module.encode_array(
                np.zeros((8, 7), dtype=np.float32)
            ),
        },
    )
    server._job_result()
    server._handle(
        "feedback",
        {"type": "feedback", "action_offset": 1, **_encoded_observation(9)},
    )
    server._handle(
        "predict_start",
        {
            "type": "predict_start",
            **_encoded_observation(3),
            "committed_actions": server_module.encode_array(
                np.zeros((8, 7), dtype=np.float32)
            ),
        },
    )
    server._job_result()

    model_inputs = server.policy.model_inputs
    assert [item["main_images"].shape[1] for item in model_inputs] == [1, 4, 4]
    assert model_inputs[1]["main_images"][0, :, 0, 0, 0].tolist() == [1, 1, 1, 2]
    assert model_inputs[2]["main_images"][0, :, 0, 0, 0].tolist() == [1, 1, 2, 3]
    assert all(
        9 not in item["main_images"][0, :, 0, 0, 0].tolist() for item in model_inputs
    )
    assert [
        values["video_frames"]
        for event, values in server.audit.events
        if event == "model_input"
    ] == [1, 4, 4]


def test_reset_restarts_single_frame_causal_warmup():
    server = _inference_server()
    reset = {"type": "reset", "task_description": "pick object", "seed": 0}
    server._handle("reset", reset)
    server._handle("predict_sync", {"type": "predict_sync", **_encoded_observation(1)})

    server._handle("reset", reset)
    server._handle("predict_sync", {"type": "predict_sync", **_encoded_observation(7)})

    assert [
        item["main_images"][0, :, 0, 0, 0].tolist()
        for item in server.policy.model_inputs
    ] == [[1], [7]]


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
