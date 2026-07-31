# Third-party sources

`manifest.yaml` is the source-of-record for upstream URLs, immutable revisions,
licenses and route ownership. Run `bash scripts/bootstrap/fetch_upstreams.sh`
to clone the selected revisions under `external/` (or an alternate directory
specified by `FBFM_EXTERNAL_ROOT`). Existing worktrees are never overwritten;
the script stops if local modifications would be lost.

The repository contains integration code and small compatibility patches, not
the upstream repositories themselves. Check each upstream license before
redistributing a deployment that includes its source, model weights, simulator
assets or CUDA extensions. Checkpoint and tokenizer files are intentionally not
tracked.

The RoboTwin route also fetches pinned PyTorch3D and CuRobo sources. CuRobo is
distributed under NVIDIA's source-code license with a non-commercial
research/evaluation use limitation; it is not relicensed by FBFM. RoboTwin's
large Hugging Face asset snapshot is downloaded separately by
`scripts/bootstrap/fetch_robotwin_assets.py`, verified against the revision and
three SHA256 values in `manifest.yaml`, and never included in the submission
archive.

Wan2.2 is kept as an overlay in [`wam/wan2.2`](../wam/wan2.2/). Applying
`patches/wan2.2_fbfm.patch` to a clean checkout at the pinned Wan2.2 revision
reconstructs the two upstream modifications required by the overlay.
