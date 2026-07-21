from wam.dreamzero.evaluation.robotwin.checkpoint_manifest import create_manifest


def test_checkpoint_tree_hash_is_content_addressed(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    first = create_manifest(checkpoint)
    second = create_manifest(checkpoint)
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
    (checkpoint / "config.json").write_text('{"changed": true}', encoding="utf-8")
    assert create_manifest(checkpoint)["checkpoint_sha256"] != first["checkpoint_sha256"]
