"""Детерминированная выборка целых единиц UMR перед ассесором.

sample_size ограничивает число единиц отбора: packed dialogue, сессий
или отдельных строк. Размер сессии не влияет на включение. Порядок задаётся
sha256 от seed и устойчивого ключа; исходное содержимое строк сохраняется.
"""

from __future__ import annotations

import ast
import hashlib
import heapq
import io
import json
import logging
import re
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def _fail(message: str) -> None:
    raise ValueError(f"[Extract Sample] {message}")


def _load_df(value: object) -> pd.DataFrame:
    """Вход порта dataframe: сам DataFrame, parquet-байты или путь артефакта."""
    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        _fail("вход monitoring_umr не подключён")
    if isinstance(value, (bytes, bytearray)):
        try:
            return pd.read_parquet(io.BytesIO(value))
        except Exception as exc:
            raise ValueError(
                "[Extract Sample] байты monitoring_umr должны содержать parquet"
            ) from exc
    path = Path(str(value))
    if path.is_dir():
        candidates = sorted(
            item for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in {".parquet", ".csv"}
        )
        if len(candidates) != 1:
            _fail(f"в каталоге monitoring_umr ожидался один файл, найдено: {candidates}")
        path = candidates[0]
    if not path.is_file():
        _fail(f"monitoring_umr не является DataFrame/parquet/путём: {value!r}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _parse_int(value: object, parameter_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"-?[0-9]+", str(value).strip()):
        _fail(f"{parameter_name} должен быть целым числом, получено: {value!r}")
    parsed = int(str(value).strip())
    if parsed < minimum:
        _fail(f"{parameter_name} должен быть >= {minimum}, получено: {parsed}")
    return parsed


def _parse_bool(value: object, parameter_name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    _fail(f"{parameter_name} должен быть булевым, получено: {value!r}")


def _blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _dialogue_len(cell: object, row_number: int) -> int:
    if isinstance(cell, str):
        try:
            cell = ast.literal_eval(cell)
        except (SyntaxError, ValueError):
            _fail(f"dialogue в строке {row_number} не разбирается как список turns")
    if isinstance(cell, (list, tuple)) or hasattr(cell, "__len__") and not isinstance(cell, str):
        return max(len(cell), 1)
    _fail(f"dialogue в строке {row_number} должен быть списком turns")


def _sampling_units(
    frame: pd.DataFrame, whole_sessions: bool
) -> tuple[list[tuple[str, list[int], int]], str]:
    """Единицы отбора: (ключ хеширования, позиции строк, число примеров) и вид
    единицы: packed_dialogue / session / row."""
    if "dialogue" in frame.columns:
        # packed: строка = сессия целиком, независимо от whole_sessions.
        units = []
        for position in range(len(frame)):
            row = frame.iloc[position]
            session = row["session_id"] if "session_id" in frame.columns else None
            if _blank(session):
                _fail(f"session_id в строке {position + 1} пуст: нужен устойчивый ключ")
            key = str(session)
            units.append((key, [position], _dialogue_len(row["dialogue"], position + 1)))
        return units, "packed_dialogue"
    if "query_id" not in frame.columns:
        _fail(
            "monitoring_umr не соответствует ни packed (нет колонки dialogue), "
            f"ни плоской форме (нет колонки query_id); колонки: {list(frame.columns)}"
        )
    group_column = next(
        (name for name in ("reference_group_id", "session_id") if name in frame.columns),
        None,
    )
    if whole_sessions and group_column is not None:
        units_by_key: dict[str, list[int]] = {}
        for position in range(len(frame)):
            group = frame.iloc[position][group_column]
            if _blank(group):
                _fail(f"{group_column} в строке {position + 1} пуст: граница группы неизвестна")
            key = str(group)
            units_by_key.setdefault(key, []).append(position)
        units = [(key, positions, len(positions)) for key, positions in units_by_key.items()]
        return units, "session"
    query_ids = frame["query_id"].tolist()
    if any(_blank(value) for value in query_ids):
        _fail("query_id содержит пустой ключ единицы отбора")
    identity_column = next(
        (name for name in ("session_id", "reference_group_id") if name in frame), None,
    )
    groups = frame[identity_column].tolist() if identity_column else [None] * len(frame)
    units = [
        (json.dumps([None if _blank(group) else str(group), str(query_id)]), [position], 1)
        for position, (group, query_id) in enumerate(zip(groups, query_ids))
    ]
    return units, "row"


def _select_positions(
    units: list[tuple[str, list[int], int]], sample_size: int, seed: int
) -> tuple[list[int], int, int]:
    """Позиции отобранных строк, число взятых примеров и число взятых единиц."""
    ranked = heapq.nsmallest(
        sample_size, units,
        key=lambda unit: (
            hashlib.sha256(f"{seed}:{unit[0]}".encode("utf-8")).digest(),
            unit[0],
        ),
    )
    selected = [position for _, positions, _ in ranked for position in positions]
    return sorted(selected), sum(examples for _, _, examples in ranked), len(ranked)


def _sample_meta(
    *, unit: str, population_units: int, population_examples: int, sampled_units: int,
    sampled_examples: int, size: int, seed: int, whole_sessions: bool, passthrough: bool,
) -> dict[str, object]:
    """Провенанс выборки для агрегатора и отчёта: что было и что отобрано."""
    return {
        "unit": unit,
        "design": "hash_srs_units_v1",
        "inclusion_probability": sampled_units / population_units if population_units else None,
        "population_units": population_units,
        "population_examples": population_examples,
        "sampled_units": sampled_units,
        "sampled_examples": sampled_examples,
        "fraction": (sampled_examples / population_examples) if population_examples else 0.0,
        "sample_size": size,
        "seed": seed,
        "whole_sessions": whole_sessions,
        "passthrough": passthrough,
    }


def main(
    monitoring_umr: object = None,
    sample_size: int = 1000,
    seed: int = 42,
    whole_sessions: bool = True,
    **_ignored: object,
) -> dict[str, object]:
    """Отдать не более sample_size целых единиц и провенанс выборки."""
    frame = _load_df(monitoring_umr)
    size = _parse_int(sample_size, "sample_size", minimum=0)
    seed_value = _parse_int(seed, "seed", minimum=-(2**63))
    keep_sessions = _parse_bool(whole_sessions, "whole_sessions")

    if frame.empty:
        logger.info("сэмплирование пропущено: строк на входе нет")
        return {
            "monitoring_umr_sample": frame,
            "sample_meta": _sample_meta(
                unit="row", population_units=0, population_examples=0, sampled_units=0,
                sampled_examples=0, size=size, seed=seed_value,
                whole_sessions=keep_sessions, passthrough=True,
            ),
        }

    units, unit_kind = _sampling_units(frame, keep_sessions)
    total_examples = sum(examples for _, _, examples in units)
    keys = [key for key, _, _ in units]
    if len(keys) != len(set(keys)):
        _fail("ключи единиц отбора должны быть уникальны")
    if size == 0 or len(units) <= size:
        logger.info(
            "единиц на входе %d, лимит %s — выборка передана целиком",
            len(units), size or "нет",
        )
        return {
            "monitoring_umr_sample": frame,
            "sample_meta": _sample_meta(
                unit=unit_kind, population_units=len(units),
                population_examples=total_examples, sampled_units=len(units),
                sampled_examples=total_examples, size=size, seed=seed_value,
                whole_sessions=keep_sessions, passthrough=True,
            ),
        }

    positions, taken_examples, taken_units = _select_positions(units, size, seed_value)
    result = frame.iloc[positions].reset_index(drop=True)
    logger.info(
        "отобрано %d примеров из %d (строк %d из %d, сессий %d из %d, seed=%d)",
        taken_examples, total_examples, len(result), len(frame),
        taken_units, len(units), seed_value,
    )
    return {
        "monitoring_umr_sample": result,
        "sample_meta": _sample_meta(
            unit=unit_kind, population_units=len(units), population_examples=total_examples,
            sampled_units=taken_units, sampled_examples=taken_examples, size=size,
            seed=seed_value, whole_sessions=keep_sessions, passthrough=False,
        ),
    }
