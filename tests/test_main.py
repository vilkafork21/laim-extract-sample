"""Тесты ноды laim-extract-sample: детерминированное сэмплирование UMR."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "extract_sample_main", Path(__file__).resolve().parents[1] / "main.py"
)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


def flat_umr(sessions: dict[str, int]) -> pd.DataFrame:
    """Плоский UMR: {session_id: число примеров}."""
    rows = []
    for session_id, count in sessions.items():
        for turn in range(1, count + 1):
            rows.append({
                "query_id": f"{session_id}-t{turn}",
                "input_query": f"вопрос {session_id}/{turn}",
                "output_answer": f"ответ {session_id}/{turn}",
                "session_id": session_id,
            })
    return pd.DataFrame(rows)


def packed_umr(sessions: dict[str, int]) -> pd.DataFrame:
    """Packed dialogue UMR: {session_id: число turn}."""
    return pd.DataFrame([
        {
            "session_id": session_id,
            "dialogue": [
                [f"{session_id}-t{turn}", f"в{turn}", f"о{turn}"]
                for turn in range(1, count + 1)
            ],
        }
        for session_id, count in sessions.items()
    ])


def test_flat_whole_sessions_not_cut():
    frame = flat_umr({"s1": 2, "s2": 3, "s3": 4, "s4": 2})
    result = node.main(monitoring_umr=frame, sample_size=2)["monitoring_umr_sample"]
    assert result["session_id"].nunique() == 2
    taken = result.groupby("session_id").size().to_dict()
    for session_id, count in taken.items():
        assert count == frame.groupby("session_id").size()[session_id], (
            f"сессия {session_id} разрезана"
        )
    assert list(result.columns) == list(frame.columns)


def test_packed_limit_counts_sessions():
    frame = packed_umr({"s1": 3, "s2": 3, "s3": 3, "s4": 3})
    result = node.main(monitoring_umr=frame, sample_size=2)["monitoring_umr_sample"]
    total_turns = sum(len(dialogue) for dialogue in result["dialogue"])
    assert total_turns == 6
    assert len(result) == 2


def test_packed_repr_preserves_all_turns_in_selected_sessions():
    frame = packed_umr({"s1": 3, "s2": 3, "s3": 3, "s4": 3})
    frame["dialogue"] = frame["dialogue"].map(repr)

    result = node.main(monitoring_umr=frame, sample_size=2)["monitoring_umr_sample"]

    assert sum(len(ast.literal_eval(dialogue)) for dialogue in result["dialogue"]) == 6
    assert len(result) == 2


def test_invalid_packed_dialogue_fails_closed():
    frame = pd.DataFrame({"session_id": ["s1"], "dialogue": ["not a turns list"]})

    with pytest.raises(ValueError, match="список turns"):
        node.main(monitoring_umr=frame, sample_size=1)


def test_session_length_does_not_limit_inclusion():
    frame = packed_umr({"large": 4, "small": 2})

    result = node.main(
        monitoring_umr=frame, sample_size=2, seed=0
    )["monitoring_umr_sample"]

    assert result["session_id"].tolist() == ["large", "small"]


def test_single_long_session_is_preserved():
    frame = packed_umr({"large": 4})

    result = node.main(monitoring_umr=frame, sample_size=1)["monitoring_umr_sample"]
    pd.testing.assert_frame_equal(result, frame)


def test_deterministic_and_seed_sensitive():
    frame = flat_umr({f"s{i}": 1 for i in range(20)})
    first = node.main(monitoring_umr=frame, sample_size=5, seed=1)["monitoring_umr_sample"]
    second = node.main(monitoring_umr=frame, sample_size=5, seed=1)["monitoring_umr_sample"]
    pd.testing.assert_frame_equal(first, second)
    other_seed = node.main(monitoring_umr=frame, sample_size=5, seed=2)["monitoring_umr_sample"]
    assert set(first["query_id"]) != set(other_seed["query_id"])


def test_zero_and_underflow_pass_everything():
    frame = flat_umr({"s1": 2, "s2": 3})
    for kwargs in ({"sample_size": 0}, {"sample_size": 1500}):
        result = node.main(monitoring_umr=frame, **kwargs)["monitoring_umr_sample"]
        pd.testing.assert_frame_equal(result.reset_index(drop=True), frame)


def test_rows_mode_exact_size():
    frame = flat_umr({"s1": 4, "s2": 4})
    result = node.main(
        monitoring_umr=frame, sample_size=3, whole_sessions=False
    )["monitoring_umr_sample"]
    assert len(result) == 3


def test_deployed_default_caps_large_flat_umr():
    frame = flat_umr({f"s{i}": 1 for i in range(1200)})

    result = node.main(
        monitoring_umr=frame, whole_sessions=False
    )["monitoring_umr_sample"]

    assert len(result) == 1000


def test_original_row_order_preserved():
    frame = flat_umr({f"s{i}": 1 for i in range(10)})
    result = node.main(monitoring_umr=frame, sample_size=4)["monitoring_umr_sample"]
    positions = [frame.index[frame["query_id"] == qid][0] for qid in result["query_id"]]
    assert positions == sorted(positions)


def test_empty_frame_passes_through():
    frame = pd.DataFrame()
    result = node.main(monitoring_umr=frame, sample_size=5)["monitoring_umr_sample"]
    assert result.empty


def test_invalid_parameters_rejected():
    frame = flat_umr({"s1": 2})
    for bad in (-1, "abc", True, 1.5):
        with pytest.raises(ValueError, match="sample_size"):
            node.main(monitoring_umr=frame, sample_size=bad)
    with pytest.raises(ValueError, match="seed"):
        node.main(monitoring_umr=frame, sample_size=1, seed="xyz")
    with pytest.raises(ValueError, match="monitoring_umr"):
        node.main(monitoring_umr=None, sample_size=1)


def test_unknown_format_rejected():
    frame = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="dialogue|query_id"):
        node.main(monitoring_umr=frame, sample_size=1)


def test_descriptor_matches_deployed_sampling_contract():
    descriptor = json.loads(
        (Path(__file__).resolve().parents[1] / "descriptor.json").read_text("utf-8")
    )
    ports = {port["name"]: port for port in descriptor["ports"]}
    settings = descriptor["ui"]["settings"][0]["components"][0]["config"]["components"]
    sample_size = next(item for item in settings if item["parameter"] == "sample_size")

    assert inspect.signature(node.main).parameters["sample_size"].default == 1000
    assert sample_size["defaultValue"] == 1000
    assert ports["monitoring_umr"]["shape"] == "shape_dataframe"
    assert ports["monitoring_umr_sample"]["shape"] == "shape_dataframe"


def test_sample_meta_reports_population_and_selection():
    frame = flat_umr({f"s{i}": 1 for i in range(20)})
    passthrough = node.main(monitoring_umr=frame, sample_size=50)
    assert passthrough["sample_meta"] == {
        "unit": "session", "design": "hash_srs_units_v1", "inclusion_probability": 1.0,
        "population_units": 20, "population_examples": 20,
        "sampled_units": 20, "sampled_examples": 20, "fraction": 1.0,
        "sample_size": 50, "seed": 42, "whole_sessions": True, "passthrough": True,
    }
    sampled = node.main(monitoring_umr=frame, sample_size=5, seed=7)
    meta = sampled["sample_meta"]
    assert meta["passthrough"] is False and meta["seed"] == 7
    assert meta["sampled_examples"] == len(sampled["monitoring_umr_sample"]) == 5
    assert meta["population_examples"] == 20 and meta["fraction"] == 0.25

    packed = node.main(monitoring_umr=packed_umr({"s1": 3, "s2": 3}), sample_size=0)
    assert packed["sample_meta"]["unit"] == "packed_dialogue"
    assert packed["sample_meta"]["population_examples"] == 6
    assert packed["sample_meta"]["passthrough"] is True

    empty = node.main(monitoring_umr=pd.DataFrame(), sample_size=5)
    assert empty["sample_meta"]["population_units"] == 0
    assert empty["sample_meta"]["passthrough"] is True


def test_descriptor_declares_sample_meta_port():
    descriptor = json.loads(
        (Path(__file__).resolve().parents[1] / "descriptor.json").read_text("utf-8")
    )
    ports = {port["name"]: port for port in descriptor["ports"]}
    assert ports["sample_meta"]["in"] is False
    assert ports["sample_meta"]["shape"] == "shape_model"
