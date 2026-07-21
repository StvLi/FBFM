"""Distributed websocket server for the DreamZero RoboTwin bridge."""

from __future__ import annotations

import asyncio
import logging
import pickle
import traceback
from typing import Any

import torch
import torch.distributed as dist
import websockets.asyncio.server
import websockets.frames
from openpi_client import msgpack_numpy
from tianshou.data import Batch

from .bridge import DreamZeroRoboTwinBridge
from .schema import RoboTwinSchema


LOGGER = logging.getLogger(__name__)
CONTINUE = 0
SHUTDOWN = 1
IDLE = 2
RESET = 3


def _broadcast_encoded(encoded: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if dist.get_rank() == 0:
        payload = pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
        size = torch.tensor([len(payload)], dtype=torch.int64, device="cuda")
        data = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to("cuda")
        dist.broadcast(size, src=0)
        dist.broadcast(data, src=0)
        return None
    size = torch.zeros(1, dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    data = torch.empty(int(size.item()), dtype=torch.uint8, device="cuda")
    dist.broadcast(data, src=0)
    return pickle.loads(data.cpu().numpy().tobytes())


class DistributedRoboTwinPolicy:
    def __init__(self, policy: Any, schema: RoboTwinSchema, mode: str, signal_group: dist.ProcessGroup) -> None:
        self.policy = policy
        self.schema = schema
        self.mode = mode
        self.signal_group = signal_group
        self.bridge = DreamZeroRoboTwinBridge(
            policy=policy,
            schema=schema,
            mode=mode,
            forward_fn=self._forward,
        )

    def _signal(self, value: int) -> None:
        signal = torch.tensor([value], dtype=torch.int32, device="cpu")
        dist.broadcast(signal, src=0, group=self.signal_group)

    def _forward(self, encoded: dict[str, Any]):
        self._signal(CONTINUE)
        _broadcast_encoded(encoded)
        dist.barrier()
        with torch.no_grad():
            result = self.policy.lazy_joint_forward_causal(Batch(obs=encoded))
        dist.barrier()
        return result

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        endpoint = request.pop("endpoint", None)
        if endpoint == "reset":
            request["reset"] = True
        if request.get("reset"):
            response = self.bridge.reset(request.get("prompt") or request.get("task"))
            self._signal(RESET)
            return response
        return self.bridge.handle(request)


async def worker_loop(policy: Any, schema: RoboTwinSchema, mode: str, signal_group: dist.ProcessGroup) -> None:
    policy.set_fbfm_mode(mode)
    policy.set_fbfm_execution_steps(schema.execute_steps)
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    LOGGER.info("DreamZero RoboTwin worker rank %s ready", dist.get_rank())
    while True:
        dist.broadcast(signal, src=0, group=signal_group)
        value = int(signal.item())
        if value == SHUTDOWN:
            return
        if value == IDLE:
            continue
        if value == RESET:
            policy.reset_inference_session()
            continue
        if value != CONTINUE:
            raise RuntimeError(f"unknown DreamZero worker signal {value}")
        encoded = _broadcast_encoded()
        dist.barrier()
        with torch.no_grad():
            policy.lazy_joint_forward_causal(Batch(obs=encoded))
        dist.barrier()


class RoboTwinWebsocketServer:
    def __init__(
        self,
        distributed_policy: DistributedRoboTwinPolicy,
        *,
        host: str,
        port: int,
        checkpoint_sha256: str | None = None,
    ) -> None:
        self.policy = distributed_policy
        self.host = host
        self.port = port
        self.metadata = {
            "model_name": "dreamzero-robotwin",
            "embodiment": distributed_policy.schema.embodiment_tag,
            "constraint_mode": distributed_policy.mode,
            "action_horizon": distributed_policy.schema.action_horizon,
            "execute_steps": distributed_policy.schema.execute_steps,
            "checkpoint_sha256": checkpoint_sha256,
        }

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                response = self.policy.handle(request)
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                LOGGER.info("RoboTwin client disconnected")
                return
            except Exception:
                error = traceback.format_exc()
                LOGGER.exception("RoboTwin request failed")
                await websocket.send(error)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="DreamZero RoboTwin request failed",
                )
                return

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_interval=None,
        ) as server:
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self.run())
