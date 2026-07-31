import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fbfm.libero_observation import (
    LIBERO_DUMMY_ACTION,
    as_model_batch,
    extract_libero_observation,
    quaternion_xyzw_to_axis_angle,
)
from fbfm.model_runtime import build_runtime_config, reset_policy_state

FBFM_REPOSITORY = Path(__file__).resolve().parents[3]
ROUTE_REPOSITORY = FBFM_REPOSITORY / "wam" / "dreamzero-libero"


def _observation() -> dict:
    return {
        "agentview_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "robot0_eye_in_hand_image": np.arange(18, 36, dtype=np.uint8).reshape(2, 3, 3),
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.4, 0.5], dtype=np.float32),
    }


def test_libero_observation_matches_dreamzero_contract():
    source = _observation()
    converted = extract_libero_observation(source)

    np.testing.assert_array_equal(
        converted["main_image"], source["agentview_image"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        converted["wrist_image"], source["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    np.testing.assert_allclose(
        converted["state"], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5]
    )
    assert converted["state"].shape == (8,)
    assert converted["state"].dtype == np.float32
    assert converted["main_image"].flags.c_contiguous
    np.testing.assert_array_equal(
        LIBERO_DUMMY_ACTION, np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
    )

    batch = as_model_batch(converted, "put the bowl on the plate")
    assert batch["main_images"].shape == (1, 2, 3, 3)
    assert batch["wrist_images"].shape == (1, 2, 3, 3)
    assert batch["states"].shape == (1, 8)
    assert batch["task_descriptions"] == ["put the bowl on the plate"]


def test_libero_observation_validates_rotation_and_required_fields():
    np.testing.assert_allclose(
        quaternion_xyzw_to_axis_angle([0.0, 0.0, 1.0, 0.0]),
        [0.0, 0.0, np.pi],
        rtol=1e-6,
    )
    invalid = _observation()
    del invalid["robot0_eef_pos"]
    with pytest.raises(KeyError, match="robot0_eef_pos"):
        extract_libero_observation(invalid)
    with pytest.raises(ValueError, match="non-finite"):
        quaternion_xyzw_to_axis_angle([0.0, 0.0, np.nan, 1.0])


def test_runtime_config_uses_checkpoint_local_paths(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    checkpoint.mkdir()
    tokenizer.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "action_head_cfg": {
                    "config": {
                        "skip_component_loading": False,
                        "defer_lora_injection": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = build_runtime_config(checkpoint, tokenizer)

    assert config.model_path == str(checkpoint.resolve())
    assert config.tokenizer_path == str(tokenizer.resolve())
    assert config.metadata_json_path == str(
        checkpoint.resolve() / "experiment_cfg" / "metadata.json"
    )
    assert config.embodiment_tag == "libero_sim"
    assert config.precision == "bf16"
    assert config.num_action_chunks == 16
    assert config.action_head_cfg.config.skip_component_loading is True
    assert config.action_head_cfg.config.defer_lora_injection is False


def test_runtime_config_overrides_all_dreamzero_component_paths(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    wan_checkpoint = tmp_path / "Wan2.2-TI2V-5B"
    image_encoder = (
        tmp_path / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    checkpoint.mkdir()
    tokenizer.mkdir()
    wan_checkpoint.mkdir()
    text_encoder = wan_checkpoint / "models_t5_umt5-xxl-enc-bf16.pth"
    vae = wan_checkpoint / "Wan2.2_VAE.pth"
    text_encoder.touch()
    vae.touch()
    image_encoder.touch()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "diffusion_model_pretrained_path": "/example/training/wan",
                "text_encoder_pretrained_path": "/example/training/t5.pth",
                "image_encoder_pretrained_path": "/example/training/clip.pth",
                "vae_pretrained_path": "/example/training/vae.pth",
                "action_head_cfg": {
                    "config": {
                        "skip_component_loading": False,
                        "defer_lora_injection": True,
                        "diffusion_model_cfg": {
                            "diffusion_model_pretrained_path": "/example/training/wan"
                        },
                        "text_encoder_cfg": {
                            "text_encoder_pretrained_path": "/example/training/t5.pth"
                        },
                        "image_encoder_cfg": {
                            "image_encoder_pretrained_path": "/example/training/clip.pth"
                        },
                        "vae_cfg": {
                            "vae_pretrained_path": "/example/training/vae.pth"
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = build_runtime_config(
        checkpoint,
        tokenizer,
        wan_checkpoint_dir=wan_checkpoint,
        image_encoder_path=image_encoder,
    )
    action_config = config.action_head_cfg.config
    expected = {
        "diffusion": str(wan_checkpoint.resolve()),
        "text": str(text_encoder.resolve()),
        "image": str(image_encoder.resolve()),
        "vae": str(vae.resolve()),
    }
    assert config.diffusion_model_pretrained_path == expected["diffusion"]
    assert config.text_encoder_pretrained_path == expected["text"]
    assert config.image_encoder_pretrained_path == expected["image"]
    assert config.vae_pretrained_path == expected["vae"]
    assert (
        action_config.diffusion_model_cfg.diffusion_model_pretrained_path
        == expected["diffusion"]
    )
    assert (
        action_config.text_encoder_cfg.text_encoder_pretrained_path
        == expected["text"]
    )
    assert (
        action_config.image_encoder_cfg.image_encoder_pretrained_path
        == expected["image"]
    )
    assert action_config.vae_cfg.vae_pretrained_path == expected["vae"]


def test_runtime_config_fails_before_missing_components_can_download(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    wan_checkpoint = tmp_path / "Wan2.2-TI2V-5B"
    image_encoder = (
        tmp_path / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    checkpoint.mkdir()
    tokenizer.mkdir()
    wan_checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        FileNotFoundError,
        match="refusing an implicit model download",
    ) as error:
        build_runtime_config(
            checkpoint,
            tokenizer,
            wan_checkpoint_dir=wan_checkpoint,
            image_encoder_path=image_encoder,
        )

    message = str(error.value)
    assert "models_t5_umt5-xxl-enc-bf16.pth" in message
    assert "Wan2.2_VAE.pth" in message
    assert str(image_encoder.resolve()) in message


def test_model_server_requires_explicit_component_paths(tmp_path):
    script = ROUTE_REPOSITORY / "scripts" / "model_server.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-workspace",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--mode",
            "FBFM",
            "--audit",
            str(tmp_path / "audit.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--wan-checkpoint" in result.stderr
    assert "--image-encoder" in result.stderr


def test_reset_policy_state_clears_sequence_caches():
    action_head = SimpleNamespace(
        seed=None,
        language="old task",
        current_start_frame=42,
        kv_cache1=object(),
        kv_cache_neg=object(),
        crossattn_cache=object(),
        crossattn_cache_neg=object(),
        clip_feas=object(),
        ys=object(),
    )
    reset_policy_state(SimpleNamespace(action_head=action_head), seed=17)

    assert action_head.seed == 17
    assert action_head.language is None
    assert action_head.current_start_frame == 0
    assert all(
        getattr(action_head, name) is None
        for name in (
            "kv_cache1",
            "kv_cache_neg",
            "crossattn_cache",
            "crossattn_cache_neg",
            "clip_feas",
            "ys",
        )
    )


@pytest.mark.parametrize(
    ("script_name", "module_name", "attribute"),
    [
        ("model_server.py", "fbfm.model_runtime", "load_policy"),
        (
            "libero_experiment.py",
            "fbfm.libero_observation",
            "extract_libero_observation",
        ),
    ],
)
def test_entrypoint_bootstraps_root_fbfm_package(
    script_name, module_name, attribute, tmp_path
):
    script = ROUTE_REPOSITORY / "scripts" / script_name
    check = (
        "import importlib, runpy; "
        f"runpy.run_path({str(script)!r}, run_name='entrypoint_contract'); "
        f"module = importlib.import_module({module_name!r}); "
        f"assert hasattr(module, {attribute!r}), module.__file__; "
        "print(module.__file__)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", check],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(FBFM_REPOSITORY / "fbfm") in result.stdout
