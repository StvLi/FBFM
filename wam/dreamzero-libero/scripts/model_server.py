#!/usr/bin/env python3
"""Launch the DreamZero FBFM server in the Python 3.11 model environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-workspace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--mode", choices=("NONE", "RTC", "FBFM"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--state-weight", type=float, default=56 / 9600)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()

    workspace = args.base_workspace.resolve()
    sys.path[:0] = [str(workspace), str(workspace / "RLinf"), str(workspace / "dreamzero")]
    # The released LIBERO checkpoint was evaluated with DreamZero's native
    # 8-evaluation cache schedule over 16 UniPC steps. FBFM hooks those native
    # evaluations; changing this to 16 changes the zero-guidance policy itself.
    os.environ["NUM_DIT_STEPS"] = "8"
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    from fbfm.model_runtime import load_policy, reset_policy_state
    from dreamzero_fbfm.constraints import ActionNormalizer
    from dreamzero_fbfm.server import ModelServer

    model, load_report = load_policy(args.checkpoint, args.tokenizer, device=args.device)
    normalizer = ActionNormalizer.from_metadata(
        args.checkpoint / "experiment_cfg" / "metadata.json",
        model_dim=int(model.action_head.model.action_dim),
    )
    server = ModelServer(
        model,
        normalizer,
        reset_policy_state,
        mode=args.mode,
        host=args.host,
        port=args.port,
        beta=args.beta,
        state_weight=args.state_weight,
        audit_path=args.audit,
    )
    ready = {
        "status": "ready",
        "mode": args.mode,
        "host": args.host,
        "port": args.port,
        "state_weight": args.state_weight,
        "load": load_report,
    }
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ready, sort_keys=True), flush=True)
    server.serve()


if __name__ == "__main__":
    main()
