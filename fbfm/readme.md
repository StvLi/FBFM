# FBFM shared package

Maintainer: [@StvLi](https://github.com/StvLi)

This directory contains repository-wide FBFM method code and small deployment
adapters that are shared with a benchmark route. It is importable as the
namespace package `fbfm` when the repository root is on `PYTHONPATH`.

## Contents

| Path | Owner | Purpose |
| --- | --- | --- |
| `policies/fbfm/` | general/Lerobot route | `PrevChunk`, `RTCPrevChunk`, `RTCProcessor`, and `RTCConfig` |
| `model_runtime.py` | DreamZero model process | strict single-GPU loading and per-episode cache reset for the RLinf checkpoint |
| `libero_observation.py` | LIBERO simulator process | camera rotation, quaternion conversion, 8D state construction, and dummy action contract |

The general policy implementation is based on RTC in
[Hugging Face Lerobot](https://github.com/huggingface/lerobot). The two
DreamZero adapters live here because launchers import them as
`fbfm.model_runtime` and `fbfm.libero_observation`. The joint FBFM solver,
transport, pseudo-clock, benchmark runner, and tests remain route-local under
[`../wam/dreamzero-libero`](../wam/dreamzero-libero/README.md).

The model adapter requires the DreamZero/RLinf Python 3.11 environment with
PyTorch and OmegaConf. The observation adapter requires NumPy and is used from
the independent LIBERO simulator environment. Neither adapter downloads a
checkpoint or mutates upstream source.
