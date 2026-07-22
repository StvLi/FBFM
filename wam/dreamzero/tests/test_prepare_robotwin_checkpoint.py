import json

from wam.dreamzero.evaluation.robotwin.checkpoint_manifest import create_manifest, hash_file
from wam.dreamzero.evaluation.robotwin.prepare_checkpoint import prepare_deploy_checkpoint
from wam.dreamzero.evaluation.robotwin.schema import RoboTwinSchema


def _schema_template() -> dict:
    fields = [
        {"key": "state.all", "start": 0, "stop": 14},
    ]
    action_fields = [
        {"key": "action.all", "start": 0, "stop": 14},
    ]
    return {
        "embodiment_tag": "robotwin",
        "camera_order": [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
        "video_keys": ["video.high", "video.left", "video.right"],
        "state_fields": fields,
        "action_fields": action_fields,
        "state_dim": 14,
        "action_dim": 14,
        "action_horizon": 24,
        "execute_steps": 8,
        "frames_per_chunk": 4,
        "normalization_metadata": "replace-me",
        "normalization_sha256": "0" * 64,
        "action_representation": "robotwin_native_eef_position_absolute",
    }


def test_prepare_deploy_checkpoint_is_compact_and_content_addressed(tmp_path):
    source = tmp_path / "checkpoint-5000"
    experiment_cfg = source / "experiment_cfg"
    experiment_cfg.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"lora")
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "trainer_state.json").write_text('{"global_step": 5000}', encoding="utf-8")
    (source / "global_step5000").mkdir()
    (source / "global_step5000" / "optimizer.pt").write_bytes(b"huge training state")
    (experiment_cfg / "conf.yaml").write_text("save_lora_only: true\n", encoding="utf-8")
    (experiment_cfg / "metadata.json").write_text("{}", encoding="utf-8")
    stats = tmp_path / "stats.json"
    stats.write_text('{"native": true}', encoding="utf-8")
    template = tmp_path / "schema.json"
    template.write_text(json.dumps(_schema_template()), encoding="utf-8")
    assets = tmp_path / "assets.json"
    assets.write_text('{"assets": []}', encoding="utf-8")
    deploy = tmp_path / "deploy"

    manifest = prepare_deploy_checkpoint(
        source_checkpoint=source,
        deploy_checkpoint=deploy,
        normalization_metadata=stats,
        schema_template=template,
        assets_manifest=assets,
        git_commit="abc123",
    )

    assert not (deploy / "global_step5000").exists()
    assert (deploy / "model.safetensors").read_bytes() == b"lora"
    schema = RoboTwinSchema.load(deploy / "robotwin_schema.json")
    assert schema.normalization_sha256 == hash_file(stats)
    provenance = json.loads((deploy / "source_checkpoint.json").read_text(encoding="utf-8"))
    assert provenance["source_global_step"] == 5000
    assert provenance["git_commit"] == "abc123"
    assert manifest["checkpoint_path"] == str(deploy.resolve())
    assert create_manifest(deploy)["checkpoint_sha256"] == manifest["checkpoint_sha256"]
