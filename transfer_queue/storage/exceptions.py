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

"""Public exceptions shared across TransferQueue storage components."""


class StorageKeyNotFoundError(KeyError):
    """Raised when a requested global index is absent from a storage unit.

    Reads and ``clear`` are concurrent by design, so a key returned by ``kv_retrieve_meta`` can be
    cleared before ``get_data`` reaches the storage unit. Callers that tolerate that race catch this
    instead of matching on message text.
    """
