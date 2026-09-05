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

import multiprocessing
import os
import subprocess
import sys
import tempfile
import traceback

import pytest
import ray
import torch
from omegaconf import OmegaConf
from ray.cluster_utils import Cluster

import transfer_queue as tq


def _check_startup(address, num_storage_units, connection):
    try:
        ray.init(address=address, log_to_driver=False)
        conf = OmegaConf.create(
            {
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {"num_data_storage_units": num_storage_units},
                }
            }
        )
        connection.send("connected")
        if num_storage_units == 2:
            try:
                tq.init(conf)
            except RuntimeError as error:
                assert isinstance(error.__cause__, ray.exceptions.GetTimeoutError)
                assert "backend.SimpleStorage.num_data_storage_units=2" in str(error)
                connection.send("timed_out")
                conf.backend.SimpleStorage.num_data_storage_units = 1
                tq.init(conf)
            else:
                raise AssertionError("Storage startup should time out when a unit cannot acquire its CPU")
        else:
            tq.init(conf)

        tq.kv_put(key="sample", partition_id="startup", fields={"value": torch.tensor([1, 2, 3])})
        result = tq.kv_batch_get(keys="sample", partition_id="startup")
        rows = list(result["value"])
        assert len(rows) == 1
        assert torch.equal(rows[0], torch.tensor([1, 2, 3]))
        tq.kv_clear(keys="sample", partition_id="startup")
        connection.send("readback_ok")
    except Exception:
        connection.send(traceback.format_exc())
    finally:
        try:
            tq.close()
        finally:
            ray.shutdown()
            connection.close()


def _run_in_isolated_cluster(num_storage_units):
    # A separate driver lets the watchdog stop a blocked init without changing
    # another test's Ray session. This process owns the cluster cleanup.
    cluster = Cluster()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = None
    try:
        cluster.add_node(num_cpus=2, include_dashboard=False)
        process = context.Process(target=_check_startup, args=(cluster.address, num_storage_units, sender))
        process.start()
        sender.close()

        assert receiver.poll(45), "The test driver did not connect to Ray"
        message = receiver.recv()
        assert message == "connected", message
        if num_storage_units == 2:
            assert receiver.poll(90), "tq.init() stayed blocked instead of reporting the startup timeout"
            message = receiver.recv()
            assert message == "timed_out", message

        assert receiver.poll(45), "Initialization or data access did not finish"
        message = receiver.recv()
        assert message == "readback_ok", message
        process.join(timeout=15)
        assert process.exitcode == 0
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        receiver.close()
        sender.close()
        cluster.shutdown()


@pytest.mark.parametrize("num_storage_units", [1, 2], ids=["successful_startup", "timeout_then_retry"])
def test_simple_storage_startup_and_readback(num_storage_units):
    # Ray's cluster shutdown resets process-wide state. Keep cluster creation
    # and cleanup outside pytest, with separate cluster discovery files.
    with tempfile.TemporaryDirectory(dir="/tmp") as ray_temp:
        result = subprocess.run(
            [sys.executable, __file__, str(num_storage_units)],
            env={**os.environ, "RAY_TMPDIR": ray_temp},
            capture_output=True,
            text=True,
            timeout=240,
        )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run_in_isolated_cluster(int(sys.argv[1]))
