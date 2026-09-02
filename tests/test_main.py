"""Тесты ноды laim-extract-sample: детерминированное сэмплирование UMR."""

from __future__ import annotations

import importlib.util
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
    result = node.main(monitoring_umr=frame, sample_size=5)["monitoring_umr_sample"]
    assert len(result) <= 5
    taken = result.groupby("session_id").size().to_dict()
    for session_id, count in taken.items():
        assert count == frame.groupby("session_id").size()[session_id], (
            f"сессия {session_id} разрезана"
        )
    assert list(result.columns) == list(frame.columns)


def test_packed_limit_counts_turns():
    frame = packed_umr({"s1": 3, "s2": 3, "s3": 3, "s4": 3})
    result = node.main(monitoring_umr=frame, sample_size=7)["monitoring_umr_sample"]
    total_turns = sum(len(dialogue) for dialogue in result["dialogue"])
    assert total_turns <= 7
    assert 1 <= len(result) < len(frame)


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
