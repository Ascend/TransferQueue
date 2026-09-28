# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
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

"""Read independently encoded rows without scanning unrelated shard data."""

import pickle

import torch
from tensordict import NonTensorStack


def read_dump_row(file, offset: int, length: int, global_index: int, fields: list[str]) -> dict:
    """Read and validate exactly one record against its expected index and fields."""
    file.seek(offset)
    payload = file.read(length)
    if len(payload) != length:
        raise ValueError(f"Truncated dump row {global_index} in {file.name}")
    row = pickle.loads(payload)
    if row["global_index"] != global_index or set(row["fields"]) != set(fields):
        raise ValueError(f"Dump row {global_index} disagrees with the row index in {file.name}")
    return row["fields"]


def validate_dump_values(values: dict, schema: dict, source_index: int) -> None:
    """Validate persisted tensor values without changing their type or dtype."""
    for name, value in values.items():
        field = schema[name]
        if field["is_non_tensor"]:
            continue
        shape = field.get("per_sample_shapes", {}).get(source_index) if field["is_nested"] else field["shape"]
        if shape is None:
            raise ValueError(f"Dump field {name!r} has no saved shape at row {source_index}")
        actual_shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        # Existing dense scalar fields use a one-element metadata shape.
        scalar = actual_shape == () and tuple(shape) == (1,) and not field["is_nested"]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != field["dtype"]
            or (actual_shape != tuple(shape) and not scalar)
        ):
            raise ValueError(f"Dump field {name!r} disagrees with its saved schema at row {source_index}")


def select_dump_schema(schema: dict, source_indexes: list[int], target_indexes: list[int], names: tuple) -> dict:
    """Remap only the selected nested shapes to the current destination indexes."""
    selected = {}
    for name in names:
        field = dict(schema[name])
        if field["is_nested"]:
            field["per_sample_shapes"] = {
                target: field["per_sample_shapes"][source]
                for source, target in zip(source_indexes, target_indexes, strict=True)
            }
        selected[name] = field
    return selected


def pack_dump_field(values: list, schema: dict):
    """Build fallback KV batches according to the original field contract."""
    if schema["is_non_tensor"]:
        return NonTensorStack(*values)
    if schema["is_nested"]:
        return torch.nested.as_nested_tensor(values, layout=torch.jagged)
    return torch.stack(values)


class RestorePendingError(RuntimeError):
    """The controller still reserves indexes until remote restore activity is settled."""

    def __init__(self, restore_id: str):
        self.restore_id = restore_id
        super().__init__(
            f"Restore {restore_id} has an unknown outcome; run recover_data_load before retrying or clearing"
        )
