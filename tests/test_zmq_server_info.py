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

from unittest.mock import MagicMock

import pytest
import ray

from transfer_queue.utils.zmq_utils import process_zmq_server_info


@pytest.mark.parametrize("single_handler", [False, True])
def test_server_info_preserves_default_results(monkeypatch, single_handler):
    handlers = {name: MagicMock() for name in ("second", "first")}
    results = {handler.get_zmq_server_info.remote.return_value: name for name, handler in handlers.items()}

    def get(refs, timeout=None):
        return [results[ref] for ref in refs] if isinstance(refs, list) else results[refs]

    monkeypatch.setattr(ray, "get", get)

    if single_handler:
        assert process_zmq_server_info(handlers["first"]) == "first"
    else:
        assert list(process_zmq_server_info(handlers).items()) == [("second", "second"), ("first", "first")]


def test_empty_server_info_does_not_require_ray(monkeypatch):
    def unexpected_get(*args, **kwargs):
        pytest.fail("Empty handlers must not make a Ray request")

    monkeypatch.setattr(ray, "get", unexpected_get)
    assert process_zmq_server_info({}) == {}
