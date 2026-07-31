import pytest

from scripts.run_libero_benchmark import parse_task_selectors


def test_parse_task_selectors_preserves_fixed_order():
    assert parse_task_selectors(
        ["libero_spatial:1", "libero_spatial:9", "libero_object:0"]
    ) == [
        ("libero_spatial", 1),
        ("libero_spatial", 9),
        ("libero_object", 0),
    ]


@pytest.mark.parametrize(
    "value",
    ["libero_spatial", "unknown:1", "libero_object:-1"],
)
def test_parse_task_selectors_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_task_selectors([value])


def test_parse_task_selectors_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        parse_task_selectors(["libero_object:6", "libero_object:6"])
