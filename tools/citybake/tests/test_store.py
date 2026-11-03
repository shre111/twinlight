import json
from pathlib import Path

import numpy as np
import pytest

from twl_citybake.store import read_table, write_json, write_table


def sample_columns() -> dict[str, np.ndarray]:
    return {
        "zeta": np.array([3, 1, 2], dtype=np.int64),
        "alpha": np.array([0.5, 1.5, -2.25], dtype=np.float32),
        "flag": np.array([1, 0, 1], dtype=np.uint8),
        "xy": np.arange(6, dtype=np.float64).reshape(3, 2),
        "small": np.array([7, 8, 9], dtype=np.uint16),
        "signed": np.array([-1, 0, 1], dtype=np.int8),
    }


def test_round_trip_preserves_values_dtypes_and_column_order(tmp_path: Path) -> None:
    columns = sample_columns()
    write_table(tmp_path / "t", columns)

    loaded = read_table(tmp_path / "t")

    assert list(loaded) == list(columns)
    for name, expected in columns.items():
        assert loaded[name].dtype == expected.dtype
        assert loaded[name].shape == expected.shape
        np.testing.assert_array_equal(loaded[name], expected)


def test_each_column_is_one_npy_file_and_metadata_lists_rows_and_dtypes(tmp_path: Path) -> None:
    columns = sample_columns()
    write_table(tmp_path / "t", columns)

    assert sorted(p.name for p in (tmp_path / "t").iterdir()) == sorted(
        [*(f"{name}.npy" for name in columns), "_table.json"]
    )
    meta = json.loads((tmp_path / "t" / "_table.json").read_text())
    assert meta["rows"] == 3
    assert [c["name"] for c in meta["columns"]] == list(columns)
    for entry, array in zip(meta["columns"], columns.values(), strict=True):
        assert np.dtype(entry["dtype"]) == array.dtype


def test_table_metadata_has_sorted_keys(tmp_path: Path) -> None:
    write_table(tmp_path / "t", sample_columns())

    text = (tmp_path / "t" / "_table.json").read_text()

    assert text.index('"columns"') < text.index('"rows"')
    assert text.index('"dtype"') < text.index('"name"')


def test_columns_are_plain_npy_without_pickle(tmp_path: Path) -> None:
    write_table(tmp_path / "t", sample_columns())

    array = np.load(tmp_path / "t" / "alpha.npy", allow_pickle=False)

    assert array.dtype == np.float32


def test_mismatched_column_lengths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r".+"):
        write_table(tmp_path / "t", {"a": np.zeros(3), "b": np.zeros(4)})


def test_mismatch_on_first_axis_of_two_dimensional_column_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r".+"):
        write_table(tmp_path / "t", {"a": np.zeros(3), "xy": np.zeros((2, 2))})


def test_object_arrays_are_rejected(tmp_path: Path) -> None:
    with pytest.raises((ValueError, TypeError)):
        write_table(tmp_path / "t", {"a": np.array(["x", None, 3], dtype=object)})


def test_empty_table_round_trips(tmp_path: Path) -> None:
    columns = {"a": np.zeros(0, dtype=np.int64), "xy": np.zeros((0, 2), dtype=np.float32)}
    write_table(tmp_path / "t", columns)

    loaded = read_table(tmp_path / "t")

    assert loaded["a"].shape == (0,)
    assert loaded["xy"].shape == (0, 2)
    assert loaded["xy"].dtype == np.float32
    assert json.loads((tmp_path / "t" / "_table.json").read_text())["rows"] == 0


def test_rewriting_the_same_table_is_byte_identical(tmp_path: Path) -> None:
    write_table(tmp_path / "a", sample_columns())
    write_table(tmp_path / "b", sample_columns())

    names = sorted(p.name for p in (tmp_path / "a").iterdir())
    assert names == sorted(p.name for p in (tmp_path / "b").iterdir())
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_rewriting_over_an_existing_table_keeps_bytes(tmp_path: Path) -> None:
    write_table(tmp_path / "t", sample_columns())
    first = {p.name: p.read_bytes() for p in (tmp_path / "t").iterdir()}

    write_table(tmp_path / "t", sample_columns())

    assert {p.name: p.read_bytes() for p in (tmp_path / "t").iterdir()} == first


def test_write_json_matches_sorted_indented_utf8_with_trailing_newline(tmp_path: Path) -> None:
    obj = {"b": [1, 2.5, None], "a": {"z": "© OpenStreetMap", "y": True}}
    write_json(tmp_path / "x.json", obj)

    text = (tmp_path / "x.json").read_bytes().decode("utf-8")

    assert text == json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert "©" in text
    assert text.index('"a"') < text.index('"b"')


def test_write_json_is_byte_identical_for_different_key_insertion_order(tmp_path: Path) -> None:
    write_json(tmp_path / "a.json", {"x": 1, "y": {"p": 1, "q": 2}})
    write_json(tmp_path / "b.json", {"y": {"q": 2, "p": 1}, "x": 1})

    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
