"""Validated localhost JSON protocol for split DreamZero/LIBERO environments."""

from __future__ import annotations

import base64
import json
import socket
import struct
from typing import Any

import numpy as np

VERSION = 1
MAX_BYTES = 64 * 1024 * 1024
HEADER = struct.Struct("!Q")
ALLOWED_DTYPES = {"uint8", "float32"}


def encode_array(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    dtype = str(array.dtype)
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"unsupported array dtype {dtype}")
    return {
        "dtype": dtype,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_array(value: Any, *, dtype: str) -> np.ndarray:
    if not isinstance(value, dict) or value.get("dtype") != dtype:
        raise ValueError(f"expected encoded {dtype} array")
    shape = value.get("shape")
    if not isinstance(shape, list) or not all(isinstance(size, int) and 0 <= size <= 4096 for size in shape):
        raise ValueError(f"invalid array shape {shape}")
    try:
        raw = base64.b64decode(value.get("data", ""), validate=True)
    except Exception as error:
        raise ValueError("invalid base64 array") from error
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if len(raw) != expected:
        raise ValueError(f"array payload has {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _receive_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise ConnectionError("connection closed during a message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_message(connection: socket.socket) -> dict[str, Any] | None:
    header = _receive_exact(connection, HEADER.size)
    if header is None:
        return None
    size = HEADER.unpack(header)[0]
    if not 0 < size <= MAX_BYTES:
        raise ValueError(f"message size {size} is invalid")
    payload = _receive_exact(connection, size)
    if payload is None:
        raise ConnectionError("connection closed before payload")
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict) or decoded.get("version") != VERSION:
        raise ValueError("invalid protocol message")
    return decoded


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    body = {**message, "version": VERSION}
    payload = json.dumps(body, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_BYTES:
        raise ValueError("message exceeds protocol limit")
    connection.sendall(HEADER.pack(len(payload)) + payload)
