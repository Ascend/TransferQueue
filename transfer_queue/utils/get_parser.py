# Copyright 2026 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Get parsers: callables a SimpleStorage unit applies to stored values before replying to ``kv_batch_get``.

A parser trims a (key, field) value to the part the caller needs, so only that part is serialized and sent."""

import reprlib
from collections.abc import Callable, Sequence
from typing import Any

from tensordict import NonTensorData, NonTensorStack

GetParser = Callable[[Any, Any], Any]
"""``parser(value, parser_arg) -> parsed value``, called once per selected (key, field)."""


def select_list_items(value: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    """Return the items of a list-valued field at ``indices``, in that order, e.g. some images of an image list.

    An index outside ``[0, len(value))`` raises ``IndexError`` instead of wrapping around."""
    if not isinstance(value, list | tuple):
        raise TypeError(f"select_list_items needs a list or tuple value, got {type(value).__name__}")
    out_of_range = [index for index in indices if not 0 <= index < len(value)]
    if out_of_range:
        raise IndexError(f"indices {out_of_range} are out of range for a list of {len(value)} items")
    return [value[index] for index in indices]


def validate_get_parser(parser: GetParser | None, parser_args: Sequence[Any] | None, num_keys: int) -> None:
    """Check a ``kv_batch_get`` parser request before anything is sent to the storage units."""
    if parser is None:
        if parser_args is not None:
            raise ValueError("parser_args was given without a parser; pass parser= as well or drop parser_args")
        return
    if not callable(parser):
        raise TypeError(f"parser must be callable, got {type(parser).__name__}")
    if parser_args is not None and len(parser_args) != num_keys:
        raise ValueError(
            f"parser_args has {len(parser_args)} entries but {num_keys} keys are fetched; give one per key"
        )


def apply_get_parser(
    field_data: dict[str, list[Any]],
    parser: GetParser,
    parser_args: Sequence[Any] | None,
    global_indexes: Sequence[int],
) -> dict[str, list[Any]]:
    """Replace every stored value by ``parser(value, parser_arg)`` of its key; values arrive unwrapped from NonTensor*.

    A parser error is re-raised as ``ValueError`` naming the field, global index and parser_arg it failed on."""
    if parser_args is None:
        parser_args = [None] * len(global_indexes)
    elif len(parser_args) != len(global_indexes):
        raise ValueError(f"got {len(parser_args)} parser_args for {len(global_indexes)} global indexes")
    parser_name = getattr(parser, "__qualname__", type(parser).__name__)
    parsed: dict[str, list[Any]] = {}
    for field, values in field_data.items():
        column = []
        for value, parser_arg, global_index in zip(values, parser_args, global_indexes, strict=True):
            try:
                column.append(parser(_unwrap_non_tensor(value), parser_arg))
            except Exception as e:
                raise ValueError(
                    f"get parser {parser_name} failed on field {field!r} of global index {global_index} "
                    f"with parser_arg {reprlib.repr(parser_arg)}: {type(e).__name__}: {e}"
                ) from e
        parsed[field] = column
    return parsed


def _unwrap_non_tensor(value: Any) -> Any:
    if isinstance(value, NonTensorStack):
        return value.tolist()
    if isinstance(value, NonTensorData):
        return value.data
    return value
