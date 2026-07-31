import asyncio
from pathlib import Path
import sys
import time

import numpy as np
import pytest


LINGBOT_VA_ROOT = Path(__file__).resolve().parents[1]
WAN_VA_ROOT = LINGBOT_VA_ROOT / "wan_va"
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_VA_ROOT))

from utils.Simple_Remote_Infer.deploy.msgpack_numpy import Packer, unpackb
from utils.Simple_Remote_Infer.deploy.websocket_policy_server import (
    WebsocketPolicyServer,
)
from utils.sever_utils import DistributedModelWrapper
from evaluation.robotwin.pseudo_async import PseudoAsyncHistory, solver_step_grant


class _SlowPolicy:
    def infer(self, obs):
        if obs.get("feedback"):
            return {"feedback_queued": True}
        time.sleep(0.25)
        return {"action": "done"}


def test_websocket_feedback_remains_responsive_during_inference():
    async def exercise():
        import websockets.asyncio.client
        import websockets.asyncio.server

        policy_server = WebsocketPolicyServer(_SlowPolicy(), host="127.0.0.1", port=0)
        async with websockets.asyncio.server.serve(
            policy_server._handler,
            "127.0.0.1",
            0,
            compression=None,
            max_size=None,
            ping_interval=None,
        ) as server:
            port = server.sockets[0].getsockname()[1]
            uri = f"ws://127.0.0.1:{port}"
            async with (
                websockets.asyncio.client.connect(uri, proxy=None) as infer_ws,
                websockets.asyncio.client.connect(uri, proxy=None) as feedback_ws,
            ):
                await infer_ws.recv()
                await feedback_ws.recv()
                packer = Packer()
                await infer_ws.send(packer.pack({"infer": True}))
                infer_result = asyncio.create_task(infer_ws.recv())
                await asyncio.sleep(0.03)

                started = time.monotonic()
                await feedback_ws.send(packer.pack({"feedback": True}))
                feedback_result = unpackb(await feedback_ws.recv())
                feedback_latency = time.monotonic() - started

                assert feedback_result["feedback_queued"]
                assert feedback_latency < 0.20
                assert not infer_result.done()
                assert unpackb(await infer_result)["action"] == "done"

    asyncio.run(exercise())


def test_distributed_wrapper_queues_feedback_without_collective():
    class FakeModel:
        def __init__(self):
            self.queued = []

        def is_inference_running(self):
            return True

        def enqueue_live_feedback(self, obs):
            self.queued.append(obs)
            return {"feedback_queued": True}

    model = FakeModel()
    wrapper = DistributedModelWrapper(model, local_rank=0)
    result = wrapper.infer({"feedback": True, "obs": [1]})
    assert result == {"feedback_queued": True}
    assert model.queued == [{"feedback": True, "obs": [1]}]


def test_distributed_wrapper_also_queues_feedback_between_inferences():
    class IdleModel:
        def __init__(self):
            self.queued = []

        def is_inference_running(self):
            return False

        def enqueue_live_feedback(self, obs):
            self.queued.append(obs)
            return {"feedback_queued": True}

    model = IdleModel()
    wrapper = DistributedModelWrapper(model, local_rank=0)
    result = wrapper.infer({"feedback": True, "obs": [1, 2, 3, 4]})
    assert result == {"feedback_queued": True}
    assert model.queued == [{"feedback": True, "obs": [1, 2, 3, 4]}]


def test_pseudo_async_grants_cover_solver_trajectory_exactly():
    grants = [
        solver_step_grant(
            step,
            total_simulation_steps=16,
            total_solver_steps=26,
        )
        for step in range(1, 17)
    ]
    assert sum(grants) == 26
    assert set(grants) == {1, 2}
    assert sum(grants[:4]) == 6
    assert sum(grants[:8]) == 13
    assert sum(grants[:12]) == 19


def test_pseudo_async_history_commits_each_segment_once_and_in_order():
    initial_action = np.arange(2 * 2 * 4).reshape(2, 2, 4)
    history = PseudoAsyncHistory(initial_action)

    observations, action_frames = history.take()
    assert observations == []
    assert np.array_equal(action_frames, initial_action[:, :1])
    with pytest.raises(RuntimeError, match="already been consumed"):
        history.take()

    executed_observations = ["obs-1", "obs-2", "obs-3", "obs-4"]
    history.stage_execution(
        executed_observations,
        initial_action,
        execution_horizon=4,
    )
    executed_observations.append("late-mutation")
    observations, action_frames = history.take()
    assert observations == ["obs-1", "obs-2", "obs-3", "obs-4"]
    assert np.array_equal(action_frames, initial_action[:, -1:])


def test_pseudo_async_history_rejects_unaligned_or_overwritten_segments():
    action = np.zeros((2, 2, 4))
    history = PseudoAsyncHistory(action)
    with pytest.raises(RuntimeError, match="consume pending history"):
        history.stage_execution([], action, execution_horizon=4)

    history.take()
    with pytest.raises(ValueError, match="complete Lingbot action frames"):
        history.stage_execution([], action, execution_horizon=3)
