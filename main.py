"""LAIM Extract Sample — детерминированное сэмплирование UMR перед ассесором.

Нода ставится между конвертером трейсов (OUT monitoring_umr) и LAIM Asessor
Agent (IN monitoring_umr): пропускает не более sample_size примеров, схему и
содержимое строк не меняет. Пример — turn диалога: в packed-форме (колонка
``dialogue``) это элемент списка turns, в плоской форме — строка датафрейма.

Отбор равномерный и воспроизводимый: единицы отбора ранжируются по
sha256(f"{seed}:{ключ}") и набираются по возрастанию хеша, пока влезают в
лимит целиком; сессия, не влезающая целиком, не берётся (при
whole_sessions=true диалог не режется). Результат не зависит от порядка строк
на входе и повторяется от запуска к запуску.
"""

from __future__ import annotations

import ast
import hashlib
import io
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
) -> list[tuple[str, list[int], int]]:
    """Единицы отбора: (ключ хеширования, позиции строк, число примеров)."""
    if "dialogue" in frame.columns:
        # packed: строка = сессия целиком, независимо от whole_sessions.
        units = []
        for position in range(len(frame)):
            row = frame.iloc[position]
            session = row["session_id"] if "session_id" in frame.columns else None
            key = f"row-{position}" if _blank(session) else str(session)
            units.append((key, [position], _dialogue_len(row["dialogue"], position + 1)))
        return units
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
            key = f"row-{position}" if _blank(group) else str(group)
            units_by_key.setdefault(key, []).append(position)
        return [(key, positions, len(positions)) for key, positions in units_by_key.items()]
    return [
        (
            f"row-{position}"
            if _blank(frame.iloc[position]["query_id"])
            else str(frame.iloc[position]["query_id"]),
            [position],
            1,
        )
        for position in range(len(frame))
    ]


def _select_positions(
    units: list[tuple[str, list[int], int]], sample_size: int, seed: int
) -> tuple[list[int], int, int]:
    """Позиции отобранных строк, число взятых примеров и число взятых единиц."""
    ranked = sorted(
        units,
        key=lambda unit: (
            hashlib.sha256(f"{seed}:{unit[0]}".encode("utf-8")).hexdigest(),
            unit[0],
        ),
    )
    selected: list[int] = []
    taken_examples = 0
    taken_units = 0
    budget = sample_size
    for _, positions, examples in ranked:
        if examples > budget:
            continue
        selected.extend(positions)
        budget -= examples
        taken_examples += examples
        taken_units += 1
    return sorted(selected), taken_examples, taken_units


def main(
    monitoring_umr: object = None,
    sample_size: int = 1000,
    seed: int = 42,
    whole_sessions: bool = True,
    **_ignored: object,
) -> dict[str, pd.DataFrame]:
    """Отдать ассесору не более sample_size примеров исходного UMR."""
    frame = _load_df(monitoring_umr)
    size = _parse_int(sample_size, "sample_size", minimum=0)
    seed_value = _parse_int(seed, "seed", minimum=-(2**63))
    keep_sessions = _parse_bool(whole_sessions, "whole_sessions")

    if frame.empty or size == 0:
        logger.info(
            "сэмплирование пропущено: строк на входе %d, sample_size=%s",
            len(frame), size or "все",
        )
        return {"monitoring_umr_sample": frame}

    units = _sampling_units(frame, keep_sessions)
    total_examples = sum(examples for _, _, examples in units)
    if total_examples <= size:
        logger.info(
            "примеров на входе %d <= лимита %d — выборка передана целиком",
            total_examples, size,
        )
        return {"monitoring_umr_sample": frame}

    positions, taken_examples, taken_units = _select_positions(units, size, seed_value)
    result = frame.iloc[positions].reset_index(drop=True)
    logger.info(
        "отобрано %d примеров из %d (строк %d из %d, сессий %d из %d, seed=%d)",
        taken_examples, total_examples, len(result), len(frame),
        taken_units, len(units), seed_value,
    )
    if not positions:
        _fail(
            f"лимит {size} меньше самой маленькой целой сессии"
        )
    return {"monitoring_umr_sample": result}
