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

A parser runs after the storage unit has loaded and decoded the stored value. It trims that value
to the part the caller needs; only the parsed subset is re-serialized and sent on the wire
(bandwidth). With Hybrid/SSD offload (and similar), the full stored blob may still be read from
disk and decoded before parsing — ``parser_args`` alone do not reduce disk I/O.

One get uses one ``parser`` and one per-key ``parser_arg`` for every selected field. Heterogeneous
field parsing (different parsers or args per field) needs separate gets. Pair a parser with
``select_fields`` limited to compatible fields; omitting ``select_fields`` is only valid when the
keys have a single field."""

import reprlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from tensordict import NonTensorData, NonTensorStack

GetParser = Callable[[Any, Any], Any]
"""``parser(value, parser_arg) -> parsed value``, called once per selected (key, field)."""


def select_list_items(value: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    """Return the items of a list-valued field at ``indices`` (a ``Sequence[int]``), in that order.

    Reorder and repeats are allowed (e.g. ``[3, 0, 3]``). An index outside ``[0, len(value))``
    raises ``IndexError`` instead of wrapping around. This is not the same API as ``slice_tensor``,
    which takes a dim-0 prefix length, ``slice``, or indexing ``tuple`` — not a list of indices.
    """
    if not isinstance(value, list | tuple):
        raise TypeError(f"select_list_items needs a list or tuple value, got {type(value).__name__}")
    out_of_range = [index for index in indices if not 0 <= index < len(value)]
    if out_of_range:
        raise IndexError(f"indices {out_of_range} are out of range for a list of {len(value)} items")
    return [value[index] for index in indices]


def _indexing_tuple_from_list(index: list) -> tuple[Any, ...] | None:
    """Restore a multi-dim indexing tuple that msgpack decoded as a list, or return None.

    Untyped msgpack turns ``(slice(None), slice(0, 2))`` into ``[slice(None), slice(0, 2)]``.
    Accept that shape so RPC ``parser_args`` still work; keep rejecting a pure ``list[int]``
    (that is ``select_list_items`` advanced-index style, not ``value[i, j, ...]``).
    """
    if not index:
        return None
    has_slice = False
    for item in index:
        if isinstance(item, bool):
            return None
        if isinstance(item, slice):
            has_slice = True
        elif isinstance(item, int) or item is None or item is Ellipsis:
            continue
        else:
            return None
    return tuple(index) if has_slice else None


def slice_tensor(value: torch.Tensor, index: int | slice | tuple[Any, ...]) -> torch.Tensor:
    """Return ``value`` sliced by ``index`` so only that logical content is serialized and sent on get.

    ``index`` is an ``int`` dim-0 prefix length (``value[:index]``), a ``slice``, or an indexing
    ``tuple`` applied as ``value[index]``. It is not a list of indices like ``select_list_items``;
    pass ``[0, 2]`` and you get ``TypeError``, not advanced indexing. After RPC, msgpack may
    deliver an indexing tuple as a list that still contains at least one ``slice``; that form is
    restored to a tuple. Wire size follows the sliced logical content in all cases.

    Encode without ``.contiguous()`` materialization only when the sliced result ``is_contiguous()``
    (typical: dim-0 prefix ``[:n]`` of a C-contiguous tensor). Non-contiguous slices such as
    ``[:, :k]`` or ``[::2]`` still round-trip correctly, but ``MsgpackEncoder`` may call
    ``.contiguous()`` before extracting the buffer.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"slice_tensor needs a torch.Tensor value, got {type(value).__name__}")
    if value.is_nested:
        raise TypeError("slice_tensor does not support nested tensors; pass a regular tensor field")
    if isinstance(index, bool):
        raise TypeError("slice_tensor index must be an int stop, a slice, or a tuple of slices/ints, not bool")
    if isinstance(index, int):
        if value.dim() == 0:
            raise IndexError("slice_tensor cannot take a prefix length of a 0-dim tensor")
        if not 0 <= index <= value.size(0):
            raise IndexError(f"prefix length {index} is out of range for dim 0 of size {value.size(0)}")
        return value[:index]
    if isinstance(index, list):
        # msgpack loses tuple vs list; only lists with a slice are treated as indexing tuples.
        restored = _indexing_tuple_from_list(index)
        if restored is None:
            raise TypeError(
                "slice_tensor does not take a list of indices (use select_list_items for that); "
                "pass an int stop, a slice, or a tuple of slices/ints"
            )
        index = restored
    if isinstance(index, slice | tuple):
        return value[index]
    raise TypeError(
        f"slice_tensor index must be an int stop, a slice, or a tuple of slices/ints, got {type(index).__name__}"
    )


def _coerce_int_like(arg: Any, label: str) -> int:
    """Turn a 0-d / single-element integer tensor or numpy scalar into a Python ``int``."""
    if isinstance(arg, torch.Tensor):
        if arg.dim() > 1 or arg.numel() != 1:
            raise ValueError(f"{label} must be a 0-d or single-element integer tensor, got shape {tuple(arg.shape)}")
        if arg.dtype == torch.bool or torch.is_floating_point(arg) or torch.is_complex(arg):
            raise TypeError(f"{label} must be an integer tensor, got dtype {arg.dtype}")
        return int(arg.item())
    if isinstance(arg, np.ndarray):
        if arg.ndim > 1 or arg.size != 1:
            raise ValueError(f"{label} must be a 0-d or single-element integer array, got shape {tuple(arg.shape)}")
        if not np.issubdtype(arg.dtype, np.integer):
            raise TypeError(f"{label} must be an integer array, got dtype {arg.dtype}")
        return int(arg.item())
    if isinstance(arg, np.integer):
        return int(arg)
    raise TypeError(f"{label} must be an integer tensor/array/scalar, got {type(arg).__name__}")


def _coerce_parser_args(parser_args: Any, num_keys: int) -> Sequence[Any]:
    """Coerce ``slice_tensor`` length args from tensor/numpy to ``int`` / ``list[int]``.

    Callers often pass ``attention_mask.sum(-1)`` or ``np.array(lengths)``; those must become
    Python ints before the get request is sent so storage units never see a Tensor as a slice index.
    """
    if isinstance(parser_args, torch.Tensor | np.ndarray):
        if parser_args.ndim == 0 and num_keys == 1:
            return [_coerce_int_like(parser_args, "parser_args")]
        if parser_args.ndim != 1:
            raise ValueError(
                f"parser_args tensor/array must be 1-d with length {num_keys}, got shape {tuple(parser_args.shape)}"
            )
        if len(parser_args) != num_keys:
            raise ValueError(
                f"parser_args has {len(parser_args)} entries but {num_keys} keys are fetched; give one per key"
            )
        return [_coerce_int_like(item, f"parser_args[{i}]") for i, item in enumerate(parser_args)]

    if not isinstance(parser_args, Sequence) or isinstance(parser_args, str | bytes):
        raise TypeError(
            f"parser_args must be a sequence of per-key args (or a 1-d integer tensor/array), "
            f"got {type(parser_args).__name__}"
        )

    coerced: list[Any] = []
    changed = False
    for i, arg in enumerate(parser_args):
        if isinstance(arg, np.integer) or isinstance(arg, torch.Tensor | np.ndarray):
            coerced.append(_coerce_int_like(arg, f"parser_args[{i}]"))
            changed = True
        else:
            coerced.append(arg)
    return coerced if changed else parser_args


def validate_get_parser(
    parser: GetParser | None, parser_args: Sequence[Any] | None, num_keys: int
) -> Sequence[Any] | None:
    """Check a ``kv_batch_get`` parser request before anything is sent to the storage units.

    When ``parser is slice_tensor``, returns ``parser_args`` with tensor/numpy length args coerced to
    ``int`` / ``list[int]`` (see ``_coerce_parser_args``). Other parsers leave ``parser_args``
    unchanged. Callers must use the return value.
    """
    if parser is None:
        if parser_args is not None:
            raise ValueError("parser_args was given without a parser; pass parser= as well or drop parser_args")
        return None
    if not callable(parser):
        raise TypeError(f"parser must be callable, got {type(parser).__name__}")
    if parser_args is not None:
        if parser is slice_tensor:
            parser_args = _coerce_parser_args(parser_args, num_keys)
        if len(parser_args) != num_keys:
            raise ValueError(
                f"parser_args has {len(parser_args)} entries but {num_keys} keys are fetched; give one per key"
            )
    return parser_args


def check_parser_select_fields(
    parser: GetParser | None,
    select_fields: list[str] | str | None,
    field_names: Sequence[str],
) -> None:
    """Fail early when a parser would run on every field of a multi-field get without ``select_fields``.

    One get applies one parser to every selected field; omitting ``select_fields`` is allowed only
    when ``field_names`` has at most one field. Heterogeneous parsing needs separate gets.
    """
    if parser is None or select_fields is not None:
        return
    if len(field_names) > 1:
        raise ValueError(
            "kv_batch_get with a parser omits select_fields but the keys have multiple fields "
            f"({list(field_names)}); one get applies one parser to every selected field. "
            "Pass select_fields limited to compatible fields, or issue separate gets for "
            "heterogeneous field parsing"
        )


def apply_get_parser(
    field_data: dict[str, list[Any]],
    parser: GetParser,
    parser_args: Sequence[Any] | None,
    global_indexes: Sequence[int],
) -> dict[str, list[Any]]:
    """Replace every already-loaded value by ``parser(value, parser_arg)`` of its key.

    Values arrive unwrapped from NonTensor*. Call this only after the storage unit has decoded
    the stored value (including any SSD read). The reply then serializes the parsed subset;
    disk I/O for the full blob is unchanged.
    Parser errors raise ``ValueError`` (field, global index, arg); kv_batch_get callers get ``RuntimeError``."""
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
