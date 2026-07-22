"""Create a content-addressed manifest for a DreamZero checkpoint directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXCLUDED_DIRS = {"real_world_eval_gen", "runs", "logs", "__pycache__"}
EXCLUDED_FILES = {"checkpoint_manifest.json"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(checkpoint: Path) -> dict:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    files = []
    for path in sorted(checkpoint.rglob("*")):
        relative_path = path.relative_to(checkpoint)
        if (
            not path.is_file()
            or path.name in EXCLUDED_FILES
            or any(part in EXCLUDED_DIRS for part in relative_path.parts)
        ):
            continue
        relative = relative_path.as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": hash_file(path)})
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {checkpoint}")
    tree = hashlib.sha256()
    for item in files:
        tree.update(f"{item['sha256']} {item['size']} {item['path']}\n".encode("utf-8"))
    return {
        "checkpoint_path": os.fspath(checkpoint),
        "checkpoint_sha256": tree.hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_manifest(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(manifest["checkpoint_sha256"])


if __name__ == "__main__":
    main()
