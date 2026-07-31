"""Check that streaming feedback VAE slots match native full-video encoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wan.fbfm.feedback import NativeWanStreamingEncoder
from wan.modules.vae2_2 import Wan2_2_VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    checkpoint = Path(args.ckpt_dir) / "Wan2.2_VAE.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    video = (
        torch.rand(
            3,
            9,
            args.height,
            args.width,
            generator=generator,
            device=device,
        )
        .mul_(2)
        .sub_(1)
    )
    vae = Wan2_2_VAE(vae_pth=str(checkpoint), device=device)

    native = vae.encode([video])[0]
    streaming = NativeWanStreamingEncoder(vae)
    slots = [streaming.prime(video[:, :1])]
    slots.append(streaming.encode(video[:, 1:5]))
    slots.append(streaming.encode(video[:, 5:9]))
    reconstructed = torch.cat(slots, dim=1)

    maximum_error = float((native - reconstructed).abs().max().item())
    result = {
        "native_shape": list(native.shape),
        "streaming_shape": list(reconstructed.shape),
        "maximum_absolute_error": maximum_error,
        "atol": args.atol,
        "passed": maximum_error <= args.atol,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise AssertionError(
            f"streaming feedback VAE differs from native encoding by {maximum_error}"
        )


if __name__ == "__main__":
    main()
