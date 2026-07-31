"""Thread-safe JSONL experiment records."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class JsonlAudit:
    def __init__(self, path: str | Path | None) -> None:
        self.path = None if path is None else Path(path)
        self._lock = threading.Lock()

    def write(self, event: str, **values: Any) -> None:
        if self.path is None:
            return
        record = {"event": event, "monotonic_seconds": time.monotonic(), **values}
        rendered = json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(rendered)
