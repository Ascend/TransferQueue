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

"""TQ_RAY_NAMESPACE moves the named actors out of the cluster-wide ``transfer_queue`` namespace."""

import pytest
import ray
from omegaconf import OmegaConf

import transfer_queue as tq
from transfer_queue.utils.common import ray_namespace


@pytest.mark.parametrize("value", [None, ""])
def test_default_namespace_is_shared(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TQ_RAY_NAMESPACE", raising=False)
    else:
        monkeypatch.setenv("TQ_RAY_NAMESPACE", value)
    assert ray_namespace() == "transfer_queue"


def test_env_overrides_namespace(monkeypatch):
    monkeypatch.setenv("TQ_RAY_NAMESPACE", "job-a")
    assert ray_namespace() == "job-a"


def test_controller_lives_in_configured_namespace(monkeypatch):
    monkeypatch.setenv("TQ_RAY_NAMESPACE", "job-a")
    ray.init(ignore_reinit_error=True, num_cpus=4)
    try:
        tq.init(OmegaConf.create({"backend": {"SimpleStorage": {"num_data_storage_units": 1}}}))
        try:
            ray.get_actor("TransferQueueController", namespace="job-a")
            with pytest.raises(ValueError):
                ray.get_actor("TransferQueueController", namespace="transfer_queue")
        finally:
            tq.close()
    finally:
        ray.shutdown()
