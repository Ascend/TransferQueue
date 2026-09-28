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

from collections import defaultdict
from typing import NamedTuple


class RoutingGroup(NamedTuple):
    """Indexes assigned to a storage unit and their positions in the input batch."""

    global_indexes: list[int]
    batch_positions: list[int]


def group_by_storage_unit(global_indexes: list[int], storage_unit_ids: list[str]) -> dict[str, RoutingGroup]:
    """Route indexes using the ordered unit list shared by storage and restore reservations."""
    gi_lists: dict[str, list[int]] = defaultdict(list)
    pos_lists: dict[str, list[int]] = defaultdict(list)
    for pos, global_idx in enumerate(global_indexes):
        key = storage_unit_ids[global_idx % len(storage_unit_ids)]
        gi_lists[key].append(global_idx)
        pos_lists[key].append(pos)
    return {key: RoutingGroup(gi_lists[key], pos_lists[key]) for key in gi_lists}
