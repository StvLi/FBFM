"""Build a compact, content-addressed RoboTwin inference checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .checkpoint_manifest import create_manifest, hash_file
from .schema import RoboTwinSchema


REQUIRED_SOURCE_FILES = (
    "model.safetensors",
    "config.json",
    "experiment_cfg/conf.yaml",
    "experiment_cfg/metadata.json",
)


def _read_global_step(source_checkpoint: Path) -> int | None:
    state_path = source_checkpoint / "trainer_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    value = state.get("global_step")
    return None if value is None else int(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prepare_deploy_checkpoint(
    *,
    source_checkpoint: Path,
    deploy_checkpoint: Path,
    normalization_metadata: Path,
    schema_template: Path,
    assets_manifest: Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Copy only inference files, bind native stats, and create a stable manifest."""

    source_checkpoint = source_checkpoint.resolve()
    deploy_checkpoint = deploy_checkpoint.resolve()
    normalization_metadata = normalization_metadata.resolve()
    schema_template = schema_template.resolve()
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(source_checkpoint)
    for relative in REQUIRED_SOURCE_FILES:
        path = source_checkpoint / relative
        if not path.is_file():
            raise FileNotFoundError(f"source checkpoint is missing {relative}: {path}")
    if not normalization_metadata.is_file():
        raise FileNotFoundError(normalization_metadata)
    if not schema_template.is_file():
        raise FileNotFoundError(schema_template)
    if assets_manifest is not None:
        assets_manifest = assets_manifest.resolve()
        if not assets_manifest.is_file():
            raise FileNotFoundError(assets_manifest)
    if deploy_checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite deploy checkpoint: {deploy_checkpoint}")

    deploy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{deploy_checkpoint.name}.tmp-",
            dir=deploy_checkpoint.parent,
        )
    )
    try:
        shutil.copy2(source_checkpoint / "model.safetensors", temporary / "model.safetensors")
        shutil.copy2(source_checkpoint / "config.json", temporary / "config.json")
        shutil.copytree(source_checkpoint / "experiment_cfg", temporary / "experiment_cfg")

        stats_name = "relative_stats_dreamzero.json"
        shutil.copy2(normalization_metadata, temporary / stats_name)
        if assets_manifest is not None:
            shutil.copy2(assets_manifest, temporary / "assets_manifest.json")

        schema = json.loads(schema_template.read_text(encoding="utf-8"))
        schema["normalization_metadata"] = stats_name
        schema["normalization_sha256"] = hash_file(temporary / stats_name)
        _write_json(temporary / "robotwin_schema.json", schema)
        RoboTwinSchema.load(temporary / "robotwin_schema.json")

        provenance = {
            "source_checkpoint": os.fspath(source_checkpoint),
            "source_global_step": _read_global_step(source_checkpoint),
            "source_model_sha256": hash_file(source_checkpoint / "model.safetensors"),
            "source_config_sha256": hash_file(source_checkpoint / "config.json"),
            "normalization_source": os.fspath(normalization_metadata),
            "normalization_sha256": schema["normalization_sha256"],
            "assets_manifest_sha256": hash_file(assets_manifest) if assets_manifest else None,
            "git_commit": git_commit,
        }
        _write_json(temporary / "source_checkpoint.json", provenance)

        manifest = create_manifest(temporary)
        manifest["checkpoint_path"] = os.fspath(deploy_checkpoint)
        _write_json(temporary / "checkpoint_manifest.json", manifest)
        temporary.replace(deploy_checkpoint)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--deploy-checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-metadata", type=Path, required=True)
    parser.add_argument("--schema-template", type=Path, required=True)
    parser.add_argument("--assets-manifest", type=Path)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    manifest = prepare_deploy_checkpoint(
        source_checkpoint=args.source_checkpoint,
        deploy_checkpoint=args.deploy_checkpoint,
        normalization_metadata=args.normalization_metadata,
        schema_template=args.schema_template,
        assets_manifest=args.assets_manifest,
        git_commit=args.git_commit,
    )
    print(manifest["checkpoint_sha256"])


if __name__ == "__main__":
    main()
