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

"""Transport-layer fault contracts for SimpleStorage payload backends."""

from __future__ import annotations


class TransferError(RuntimeError):
    """Transport, protocol, or peer-configuration failure.

    ``transient`` is ``True`` only when retrying on a fresh connection could recover;
    deterministic decode, protocol, or configuration errors keep ``transient=False``.
    """

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


class TransferTimeout(TransferError):
    """Transport request timed out."""

    def __init__(self, message: str, *, wire_bytes: int = 0):
        super().__init__(message, transient=True)
        self.wire_bytes = wire_bytes


class StorageRejected(RuntimeError):
    """Storage unit received the request and rejected it."""

    def __init__(self, message: str, *, missing_key: bool = False):
        super().__init__(message)
        self.missing_key = missing_key
