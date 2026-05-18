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

from functools import wraps
from typing import Callable


class StorageBootstrapProvider:
    _providers: dict[str, Callable] = {}

    @classmethod
    def register_provider(cls, name: str):
        """Decorator to register storage provider & returns function."""

        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            cls._providers[name.lower()] = wrapper
            return wrapper

        return decorator

    @classmethod
    def get_provider(cls, name: str) -> Callable | None:
        """Get storage provider function by name."""
        return cls._providers.get(name.lower(), None)
