from collections import Counter

import pytest

from test_main import node, packed_umr, flat_umr


def test_selection_does_not_depend_on_session_length():
    short = [(f"s{i}", [i], 1) for i in range(20)]
    unequal = [(key, positions, 10000 if i == 0 else i + 1)
               for i, (key, positions, _) in enumerate(short)]
    counts = Counter()
    for seed in range(1000):
        selected, _, taken = node._select_positions(unequal, 5, seed)
        assert selected == node._select_positions(short, 5, seed)[0]
        assert taken == len(selected) == 5
        counts.update(selected)
    # При 1000 фиксированных seeds ожидаем 250 включений каждой из 20 групп.
    assert set(counts) == set(range(20))
    assert all(180 <= count <= 320 for count in counts.values())


def test_probability_describes_sessions_and_not_realized_turn_fraction():
    frame = packed_umr({"long": 100, "short": 1})
    result = node.main(frame, sample_size=1)
    meta = result["sample_meta"]
    assert meta["design"] == "hash_srs_units_v1"
    assert meta["inclusion_probability"] == 0.5
    assert meta["fraction"] in {1 / 101, 100 / 101}
    reordered = node.main(frame.iloc[::-1], sample_size=1)
    assert result["monitoring_umr_sample"].session_id.tolist() == (
        reordered["monitoring_umr_sample"].session_id.tolist()
    )


@pytest.mark.parametrize("packed", [True, False])
def test_ambiguous_sampling_keys_are_rejected(packed):
    frame = packed_umr({"a": 1, "b": 1}) if packed else flat_umr({"a": 1, "b": 1})
    key = "session_id" if packed else "query_id"
    frame[key] = "duplicate"
    if not packed:
        frame["session_id"] = "same-session"
    with pytest.raises(ValueError, match="уникальны"):
        node.main(frame, sample_size=1, whole_sessions=False)
    frame.loc[0, key] = None
    with pytest.raises(ValueError, match="пуст"):
        node.main(frame, sample_size=1, whole_sessions=False)


def test_query_id_can_repeat_in_different_sessions():
    frame = flat_umr({"a": 1, "b": 1})
    frame["query_id"] = "q1"
    result = node.main(frame, sample_size=1, whole_sessions=False)
    assert len(result["monitoring_umr_sample"]) == 1
    assert result["sample_meta"]["population_units"] == 2
