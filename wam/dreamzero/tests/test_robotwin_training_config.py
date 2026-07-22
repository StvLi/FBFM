from pathlib import Path

import yaml


def test_robotwin_has_a_unique_transform_projector_id():
    config_path = (
        Path(__file__).parents[1]
        / "groot/vla/configs/model/dreamzero/transform/base.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping = config["embodiment_tag_to_projector_index"]

    assert mapping["robotwin"] == 33
    assert list(mapping.values()).count(mapping["robotwin"]) == 1
