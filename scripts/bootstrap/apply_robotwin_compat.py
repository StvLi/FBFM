#!/usr/bin/env python3
"""Apply the two compatibility edits required by pinned RoboTwin 2.0."""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path


class CompatibilityError(RuntimeError):
    """Raised when an installed dependency no longer matches the audited source."""


SAPIEN_REPLACEMENTS = (
    (
        'with open(urdf_file, "r") as f:',
        'with open(urdf_file, "r", encoding="utf-8") as f:',
    ),
    (
        'srdf_file = urdf_file[:-4] + "srdf"',
        'srdf_file = urdf_file[:-4] + ".srdf"',
    ),
    (
        'with open(srdf_file, "r") as f:',
        'with open(srdf_file, "r", encoding="utf-8") as f:',
    ),
)

MPLIB_REPLACEMENTS = (
    (
        "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
        "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
    ),
)


def _installed_package_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise CompatibilityError(f"installed package not found: {package}")
    roots = list(spec.submodule_search_locations)
    if len(roots) != 1:
        raise CompatibilityError(
            f"expected one installed location for {package}, found {roots}"
        )
    return Path(roots[0]).resolve()


def _rewrite_exact(
    path: Path,
    replacements: tuple[tuple[str, str], ...],
    *,
    check: bool,
    dry_run: bool,
) -> bool:
    if not path.is_file():
        raise CompatibilityError(f"required installed source file is missing: {path}")

    original = path.read_text(encoding="utf-8")
    updated = original
    changed = False
    for before, after in replacements:
        before_count = updated.count(before)
        after_count = updated.count(after)
        if before_count == 1 and after_count == 0:
            if check:
                raise CompatibilityError(f"compatibility edit is not applied in {path}")
            updated = updated.replace(before, after, 1)
            changed = True
        elif before_count == 0 and after_count == 1:
            continue
        else:
            raise CompatibilityError(
                f"unexpected source in {path}: audited pattern counts are "
                f"before={before_count}, after={after_count}"
            )

    if check:
        print(f"robotwin_compat: verified {path}")
        return False
    if not changed:
        print(f"robotwin_compat: already applied {path}")
        return False
    if dry_run:
        print(f"robotwin_compat: would patch {path}")
        return True

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.fbfm-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(updated)
        os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    print(f"robotwin_compat: patched {path}")
    return True


def apply_compatibility(
    site_packages: Path | None = None,
    *,
    check: bool = False,
    dry_run: bool = False,
) -> None:
    if site_packages is None:
        sapien_root = _installed_package_root("sapien")
        mplib_root = _installed_package_root("mplib")
    else:
        site_packages = site_packages.resolve()
        sapien_root = site_packages / "sapien"
        mplib_root = site_packages / "mplib"

    _rewrite_exact(
        sapien_root / "wrapper" / "urdf_loader.py",
        SAPIEN_REPLACEMENTS,
        check=check,
        dry_run=dry_run,
    )
    _rewrite_exact(
        mplib_root / "planner.py",
        MPLIB_REPLACEMENTS,
        check=check,
        dry_run=dry_run,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages",
        type=Path,
        help="explicit site-packages root (normally auto-detected)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="verify edits without modifying files"
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="show edits without modifying files"
    )
    args = parser.parse_args()
    try:
        apply_compatibility(
            args.site_packages, check=args.check, dry_run=args.dry_run
        )
    except CompatibilityError as exc:
        parser.exit(1, f"robotwin_compat: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
