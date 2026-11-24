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

import json
import pickle
from pathlib import Path

import torch

from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.serial_utils import validate_metadata_dicts

logger = get_logger(__name__)

_METADATA_FILE = "metadata.json"
_CONTROLLER_FILE = "controller_state.pkl"


def validate_checkpoint(checkpoint_dir: str | Path) -> bool:
    """Return whether a TransferQueue checkpoint directory has a valid manifest and controller state.

    Args:
        checkpoint_dir: Path to a checkpoint directory written by ``save_checkpoint``.

    Returns:
        ``True`` if ``metadata.json`` and ``controller_state.pkl`` pass structural and
        native MessagePack metadata validation, ``False`` otherwise.
    """
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / _METADATA_FILE
    controller_path = checkpoint_dir / _CONTROLLER_FILE
    if not checkpoint_dir.is_dir() or not metadata_path.is_file() or not controller_path.is_file():
        return False

    try:
        meta = json.loads(metadata_path.read_text())
        if not isinstance(meta, dict) or not isinstance(meta.get("storage_saved"), bool):
            return False

        with open(controller_path, "rb") as f:
            state = pickle.load(f)

        if (
            not isinstance(state, dict)
            or not {"controller_id", "partitions", "index_manager", "sampler"} <= state.keys()
            or not isinstance(state["partitions"], dict)
        ):
            raise ValueError("controller checkpoint is missing required top-level keys")

        im = state["index_manager"]
        if (
            not isinstance(im, dict)
            or not {
                "partition_to_indexes",
                "reusable_indexes",
                "global_index_counter",
                "allocated_indexes",
            }
            <= im.keys()
            or not isinstance(im["reusable_indexes"], list)
            or not all(isinstance(idx, int) for idx in im["reusable_indexes"])
        ):
            raise ValueError("controller checkpoint has an invalid index_manager state")

        if not isinstance(state["sampler"], dict):
            raise ValueError("controller checkpoint has an invalid sampler state")

        for partition_id, partition in state["partitions"].items():
            if partition.custom_meta:
                validate_metadata_dicts(partition.custom_meta.values(), "custom_meta")
            if partition.field_custom_backend_meta:
                validate_metadata_dicts(partition.field_custom_backend_meta.values(), "custom_backend_meta")
            if partition.links:
                validate_metadata_dicts(partition.links.values(), "links")
            for name, field_meta in partition.field_metadata.items():
                if field_meta.dtype is not None and not isinstance(field_meta.dtype, torch.dtype):
                    raise ValueError(f"field {partition_id}.{name} has invalid dtype {field_meta.dtype!r}")
                if field_meta.shape is not None and not isinstance(field_meta.shape, tuple):
                    raise ValueError(f"field {partition_id}.{name} has non-tuple shape {field_meta.shape!r}")
                if field_meta.per_sample_shapes and any(
                    shape is not None and not isinstance(shape, tuple)
                    for shape in field_meta.per_sample_shapes.values()
                ):
                    raise ValueError(f"field {partition_id}.{name} has non-tuple per_sample_shapes")
    except Exception as e:
        logger.warning(f"Checkpoint validation failed for {checkpoint_dir}: {e}")
        return False

    return True
