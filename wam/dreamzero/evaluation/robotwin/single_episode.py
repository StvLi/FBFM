"""Run one frozen RoboTwin episode against the DreamZero websocket server."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .experiment import model_noise_seed
from .representation import eef14_to_simulator_action, format_simulator_observation


def _episode_from_environment() -> dict[str, Any]:
    value = os.environ.get("ROBOTWIN_EPISODE_JSON")
    if not value:
        raise RuntimeError("ROBOTWIN_EPISODE_JSON is required")
    episode = json.loads(value)
    for key in ("episode_id", "task", "config", "seed", "instruction", "model_noise_seed_base"):
        if key not in episode:
            raise ValueError(f"frozen episode is missing {key!r}")
    return episode


def _validate_server_metadata(metadata: dict[str, Any], expected_mode: str, checkpoint_sha256: str) -> tuple[int, int]:
    if metadata.get("constraint_mode") != expected_mode:
        raise RuntimeError(
            f"server mode {metadata.get('constraint_mode')!r} != requested mode {expected_mode!r}"
        )
    if metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("DreamZero server and episode runner use different checkpoint hashes")
    execute_steps = int(metadata["execute_steps"])
    frames_per_chunk = int(metadata["frames_per_chunk"])
    if execute_steps <= 0 or frames_per_chunk <= 0 or execute_steps % frames_per_chunk:
        raise ValueError("execute_steps must be a positive multiple of frames_per_chunk")
    return execute_steps, frames_per_chunk


def _validate_action_chunk(action: Any, execute_steps: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    expected = (14, 1, execute_steps)
    if action.shape != expected:
        raise ValueError(f"DreamZero action shape {action.shape} != {expected}")
    if not np.isfinite(action).all():
        raise ValueError("DreamZero returned a non-finite action")
    return action


def _load_robotwin_config(root: Path, episode: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    script_root = root / "script"
    for path in (root, script_root, root / "description" / "utils"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(root)

    from envs import CONFIGS_PATH

    with (root / "task_config" / f"{episode['config']}.yml").open(encoding="utf-8") as handle:
        args = yaml.safe_load(handle)
    args.update(
        {
            "task_name": episode["task"],
            "task_config": episode["config"],
            "ckpt_setting": "dreamzero_frozen_manifest",
            "policy_name": "DreamZero",
            "eval_mode": True,
            "eval_video_log": False,
        }
    )

    config_root = Path(CONFIGS_PATH)
    with (config_root / "_embodiment_config.yml").open(encoding="utf-8") as handle:
        embodiments = yaml.safe_load(handle)
    with (config_root / "_camera_config.yml").open(encoding="utf-8") as handle:
        cameras = yaml.safe_load(handle)
    camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[camera_type]["h"]
    args["head_camera_w"] = cameras[camera_type]["w"]

    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        args["left_robot_file"] = embodiments[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiments[embodiment[0]]["file_path"]
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiments[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiments[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"unsupported RoboTwin embodiment declaration {embodiment!r}")
    for side in ("left", "right"):
        robot_file = Path(args[f"{side}_robot_file"])
        with (robot_file / "config.yml").open(encoding="utf-8") as handle:
            args[f"{side}_embodiment_config"] = yaml.safe_load(handle)

    task_module = importlib.import_module(f"envs.{episode['task']}")
    task_environment = getattr(task_module, episode["task"])()
    return task_environment, args


def _verify_frozen_initialization(task_environment: Any, args: dict[str, Any], seed: int) -> None:
    from envs.utils.create_actor import UnStableError

    render_frequency = args["render_freq"]
    args["render_freq"] = 0
    try:
        task_environment.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        task_environment.play_once()
        if not (task_environment.plan_success and task_environment.check_success()):
            raise RuntimeError(f"frozen seed {seed} no longer passes the RoboTwin expert check")
    except UnStableError as exc:
        raise RuntimeError(f"frozen seed {seed} became unstable") from exc
    finally:
        task_environment.close_env()
        args["render_freq"] = render_frequency


def run_episode(*, host: str, port: int, robotwin_root: Path) -> dict[str, Any]:
    episode = _episode_from_environment()
    expected_mode = os.environ.get("FBFM_CONSTRAINT_MODE", "None")
    checkpoint_sha256 = os.environ.get("DREAMZERO_CHECKPOINT_SHA256", "")
    if not checkpoint_sha256:
        raise RuntimeError("DREAMZERO_CHECKPOINT_SHA256 is required")

    task_environment, args = _load_robotwin_config(robotwin_root, episode)
    _verify_frozen_initialization(task_environment, args, int(episode["seed"]))

    from eval_utils.policy_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=host, port=port)
    execute_steps, frames_per_chunk = _validate_server_metadata(
        client.get_server_metadata(), expected_mode, checkpoint_sha256
    )
    actions_per_feedback = execute_steps // frames_per_chunk
    instruction = str(episode["instruction"])
    chunk_index = 0
    feedback_frames = 0
    simulator_steps = 0
    success = False
    try:
        task_environment.setup_demo(now_ep_num=0, seed=int(episode["seed"]), is_test=True, **args)
        task_environment.set_instruction(instruction=instruction)
        reset = client.infer({"reset": True, "prompt": instruction})
        if not reset.get("ok"):
            raise RuntimeError(f"DreamZero reset failed: {reset}")

        while task_environment.take_action_cnt < task_environment.step_lim:
            observation = format_simulator_observation(task_environment.get_obs(), instruction)
            chunk_seed = model_noise_seed(episode, chunk_index)
            response = client.infer({"obs": observation, "inference_seed": chunk_seed})
            if response.get("mode") != expected_mode or int(response.get("inference_seed", -1)) != chunk_seed:
                raise RuntimeError("DreamZero server did not honor the requested mode/chunk noise seed")
            action = _validate_action_chunk(response["action"], execute_steps)
            key_frames = []
            for step in range(execute_steps):
                task_environment.take_action(
                    eef14_to_simulator_action(action[:, 0, step]),
                    action_type="ee",
                )
                if (step + 1) % actions_per_feedback == 0:
                    feedback = format_simulator_observation(task_environment.get_obs(), instruction)
                    key_frames.append(feedback)
                    acknowledgement = client.infer({"obs": feedback, "feedback": True})
                    if acknowledgement.get("event") != "feedback_buffered":
                        raise RuntimeError(f"DreamZero feedback failed: {acknowledgement}")
                    feedback_frames += 1
                if task_environment.take_action_cnt >= task_environment.step_lim:
                    break
            if key_frames:
                acknowledgement = client.infer({"obs": key_frames, "compute_kv_cache": True})
                if acknowledgement.get("event") != "causal_context_buffered":
                    raise RuntimeError(f"DreamZero causal context update failed: {acknowledgement}")
            chunk_index += 1
            if task_environment.eval_success:
                success = True
                break
    finally:
        try:
            simulator_steps = int(task_environment.take_action_cnt)
            task_environment.close_env(clear_cache=True)
        finally:
            client._ws.close()

    return {
        "success": bool(success),
        "seed": int(episode["seed"]),
        "instruction": instruction,
        "chunks": chunk_index,
        "feedback_frames": feedback_frames,
        "simulator_steps": simulator_steps,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN_ROOT", "/mnt/project_eai_hs/zrm/RoboTwin")),
    )
    args = parser.parse_args()
    result = run_episode(host=args.host, port=args.port, robotwin_root=args.robotwin_root.resolve())
    result_path_value = os.environ.get("ROBOTWIN_RESULT_PATH")
    if not result_path_value:
        raise RuntimeError("ROBOTWIN_RESULT_PATH is required")
    result_path = Path(result_path_value)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
