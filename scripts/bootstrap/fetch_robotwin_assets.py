#!/usr/bin/env python3
"""Download, verify, extract, and configure the pinned RoboTwin 2.0 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

DATASET_REPOSITORY = "TianxingChen/RoboTwin2.0"
DATASET_REVISION = "9dc9299c163db059931898a9f0852098a61155a1"
ARCHIVES = {
    "background_texture.zip": (
        10_970_687_027,
        "54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070",
    ),
    "embodiments.zip": (
        219_859_313,
        "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6",
    ),
    "objects.zip": (
        3_737_778_549,
        "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27",
    ),
}
SENTINELS = (
    "assets/background_texture/seen/4282.png",
    "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
    "assets/objects/001_bottle/points_info.json",
)
MANIFEST_NAME = ".fbfm-assets.json"


class AssetError(RuntimeError):
    """Raised when the downloaded asset bundle is incomplete or unsafe."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, size: int, expected_hash: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise AssetError(f"size mismatch for {path}: expected {size}, found {actual_size}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise AssetError(
            f"SHA256 mismatch for {path}: expected {expected_hash}, found {actual_hash}"
        )
    print(f"robotwin-assets: verified {path.name} ({actual_size} bytes)")


def validate_zip_members(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        name = PurePosixPath(member.filename)
        if name.is_absolute() or ".." in name.parts:
            raise AssetError(f"unsafe path in {archive.filename}: {member.filename!r}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise AssetError(f"symbolic link is not allowed in asset zip: {member.filename!r}")
        target = (destination / Path(*name.parts)).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise AssetError(
                f"asset path escapes destination in {archive.filename}: {member.filename!r}"
            ) from exc


def extract_archive(path: Path, destination: Path) -> None:
    print(f"robotwin-assets: extracting {path.name}")
    with zipfile.ZipFile(path) as archive:
        validate_zip_members(archive, destination)
        archive.extractall(destination)


def configure_embodiments(robotwin_root: Path) -> int:
    templates = sorted((robotwin_root / "assets/embodiments").rglob("*_tmp.yml"))
    if not templates:
        raise AssetError("no embodiment *_tmp.yml templates were found after extraction")
    for template in templates:
        source = template.read_text(encoding="utf-8")
        if "${ASSETS_PATH}" not in source and "$ASSETS_PATH" not in source:
            raise AssetError(f"asset template has no ASSETS_PATH placeholder: {template}")

    updater = robotwin_root / "script/update_embodiment_config_path.py"
    if not updater.is_file():
        raise AssetError(f"RoboTwin embodiment configuration script is absent: {updater}")
    try:
        subprocess.run(
            [sys.executable, str(updater)],
            cwd=robotwin_root,
            stdin=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AssetError(
            f"RoboTwin embodiment configuration script failed with exit code "
            f"{exc.returncode}: {updater}"
        ) from exc
    print(
        "robotwin-assets: configured "
        f"{len(templates)} embodiment file(s) with {updater.relative_to(robotwin_root)}"
    )
    return len(templates)


def expected_manifest() -> dict[str, object]:
    return {
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "archives": {
            name: {"bytes": size, "sha256": expected_hash}
            for name, (size, expected_hash) in ARCHIVES.items()
        },
    }


def write_manifest(robotwin_root: Path) -> None:
    path = robotwin_root / "assets" / MANIFEST_NAME
    path.write_text(
        json.dumps(expected_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"robotwin-assets: wrote provenance marker {path}")


def manifest_matches(robotwin_root: Path) -> bool:
    marker = robotwin_root / "assets" / MANIFEST_NAME
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == expected_manifest()
    except (OSError, json.JSONDecodeError):
        return False


def check_assets(robotwin_root: Path) -> None:
    missing = [
        str(robotwin_root / item)
        for item in SENTINELS
        if not (robotwin_root / item).is_file()
    ]
    if missing:
        raise AssetError("missing extracted asset sentinel(s): " + ", ".join(missing))

    templates = sorted((robotwin_root / "assets/embodiments").rglob("*_tmp.yml"))
    if not templates:
        raise AssetError("no embodiment templates found")
    marker = robotwin_root / "assets" / MANIFEST_NAME
    if not marker.is_file():
        raise AssetError(
            f"asset provenance marker is absent: {marker}; rerun without --check"
        )
    try:
        actual_manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"invalid asset provenance marker: {marker}") from exc
    if actual_manifest != expected_manifest():
        raise AssetError(f"asset provenance marker does not match this release: {marker}")
    root_text = str(robotwin_root.resolve())
    for template in templates:
        expected = template.read_text(encoding="utf-8").replace(
            "${ASSETS_PATH}", root_text
        ).replace("$ASSETS_PATH", root_text)
        target = template.with_name(template.name.replace("_tmp.yml", ".yml"))
        if not target.is_file() or target.read_text(encoding="utf-8") != expected:
            raise AssetError(f"embodiment configuration is absent or stale: {target}")
    print(
        "robotwin-assets: asset preflight passed "
        f"({len(templates)} configured embodiments at {robotwin_root})"
    )


def assets_look_complete(robotwin_root: Path) -> bool:
    return all((robotwin_root / item).is_file() for item in SENTINELS)


def default_robotwin_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    external_root = Path(os.environ.get("FBFM_EXTERNAL_ROOT", repo_root / "external"))
    return external_root / "RoboTwin"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=default_robotwin_root(),
        help="pinned RoboTwin checkout (default: $FBFM_EXTERNAL_ROOT/RoboTwin)",
    )
    parser.add_argument("--check", action="store_true", help="preflight only; never download")
    parser.add_argument("--dry-run", action="store_true", help="print the immutable download plan")
    parser.add_argument(
        "--keep-archives", action="store_true", help="retain the three verified zip files"
    )
    args = parser.parse_args()
    robotwin_root = args.robotwin_root.expanduser().resolve()
    assets_root = robotwin_root / "assets"

    if args.dry_run:
        print(f"robotwin-assets: checkout={robotwin_root}")
        print(
            "robotwin-assets: would download "
            f"dataset={DATASET_REPOSITORY} revision={DATASET_REVISION}"
        )
        for name, (size, expected_hash) in ARCHIVES.items():
            print(f"robotwin-assets:   {name} bytes={size} sha256={expected_hash}")
        print("robotwin-assets: would verify, safely extract, and configure embodiments")
        return 0

    if not (robotwin_root / ".git").exists():
        raise AssetError(f"RoboTwin Git checkout is absent: {robotwin_root}")
    if args.check:
        check_assets(robotwin_root)
        return 0

    if assets_look_complete(robotwin_root) and manifest_matches(robotwin_root):
        configure_embodiments(robotwin_root)
        check_assets(robotwin_root)
        print("robotwin-assets: existing extracted assets reused; no download needed")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AssetError(
            "huggingface_hub is unavailable; run create_envs.sh --route lingbot first"
        ) from exc

    assets_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=list(ARCHIVES),
        local_dir=str(assets_root),
    )

    archive_paths: list[Path] = []
    for name, (size, expected_hash) in ARCHIVES.items():
        path = assets_root / name
        if not path.is_file():
            raise AssetError(f"snapshot download did not produce {path}")
        verify_archive(path, size, expected_hash)
        archive_paths.append(path)
    for path in archive_paths:
        extract_archive(path, assets_root)
    configure_embodiments(robotwin_root)
    write_manifest(robotwin_root)
    check_assets(robotwin_root)

    if not args.keep_archives:
        for path in archive_paths:
            path.unlink()
        print("robotwin-assets: removed verified zip files after successful extraction")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssetError as exc:
        raise SystemExit(f"robotwin-assets: error: {exc}") from exc
