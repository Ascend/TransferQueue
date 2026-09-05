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

from unittest.mock import MagicMock, call

import pytest
import ray
from omegaconf import OmegaConf

from transfer_queue import interface
from transfer_queue.storage.bootstrap import simple_storage_bootstrap
from transfer_queue.utils import common

_NODE_A = "01" * 28
_NODE_B = "02" * 28
_NODE_C = "03" * 28
_NODE_D = "04" * 28


def _node(node_id: str, *, alive: bool = True, resources: dict[str, float] | None = None) -> dict:
    return {"NodeID": node_id, "Alive": alive, "Resources": resources or {}}


def _node_ids(strategies) -> list[str]:
    return [strategy.node_id for strategy in strategies]


def test_round_robin_uses_all_alive_nodes_by_default(monkeypatch):
    nodes = [_node(_NODE_B), _node(_NODE_C, alive=False), _node(_NODE_A)]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)

    strategies = common.get_node_round_robin_scheduling_strategies(5)

    assert _node_ids(strategies) == [_NODE_A, _NODE_B, _NODE_A, _NODE_B, _NODE_A]


def test_required_node_resource_filters_nodes_and_zero_capacity(monkeypatch):
    nodes = [
        _node(_NODE_C, resources={"storage_pool": 2}),
        _node(_NODE_B, resources={"storage_pool": 0}),
        _node(_NODE_A, resources={"storage_pool": 1}),
        _node(_NODE_D, resources={"compute_pool": 1}),
    ]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)

    strategies = common.get_node_round_robin_scheduling_strategies(4, required_node_resource="storage_pool")

    assert _node_ids(strategies) == [_NODE_A, _NODE_C, _NODE_A, _NODE_C]


def test_required_node_resource_excludes_dead_nodes(monkeypatch):
    nodes = [
        _node(_NODE_A, alive=False, resources={"storage_pool": 1}),
        _node(_NODE_B, resources={"storage_pool": 1}),
    ]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)

    strategies = common.get_node_round_robin_scheduling_strategies(2, required_node_resource="storage_pool")

    assert _node_ids(strategies) == [_NODE_B, _NODE_B]


def test_required_node_resource_raises_when_no_alive_node_matches(monkeypatch):
    nodes = [
        _node(_NODE_A, resources={"storage_pool": 0}),
        _node(_NODE_B, alive=False, resources={"storage_pool": 1}),
    ]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)

    with pytest.raises(ValueError, match="No alive Ray nodes provide custom resource 'storage_pool'"):
        common.get_node_round_robin_scheduling_strategies(1, required_node_resource="storage_pool")


def test_default_no_alive_node_error_is_unchanged(monkeypatch):
    monkeypatch.setattr(common.ray, "nodes", lambda: [])

    with pytest.raises(RuntimeError, match="No alive Ray nodes found. Is Ray initialized?"):
        common.get_node_round_robin_scheduling_strategies(1)


def test_simple_storage_initialization_forwards_required_node_resource(monkeypatch):
    strategy = MagicMock(node_id=_NODE_A)
    get_strategies = MagicMock(return_value=[strategy])
    storage_unit = MagicMock()
    storage_handle = MagicMock()
    storage_unit.options.return_value.remote.return_value = storage_handle

    monkeypatch.setattr(simple_storage_bootstrap, "get_node_round_robin_scheduling_strategies", get_strategies)
    monkeypatch.setattr(simple_storage_bootstrap, "SimpleStorageUnit", storage_unit)
    monkeypatch.setattr(simple_storage_bootstrap, "process_zmq_server_info", MagicMock(return_value={}))
    monkeypatch.setattr(ray, "available_resources", lambda: {"CPU": 1.0})

    conf = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "num_data_storage_units": 1,
                    "total_storage_size": None,
                    "required_node_resource": "storage_pool",
                },
            }
        }
    )

    handles = simple_storage_bootstrap.initialize_simple_storage(conf)

    get_strategies.assert_called_once_with(1, required_node_resource="storage_pool")
    assert handles == {"TransferQueueStorageUnit#0": storage_handle}


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_simple_storage_start_timeout_kills_units_and_reports_cpu_requirement(monkeypatch, cleanup_fails):
    strategies = [MagicMock(node_id=_NODE_A), MagicMock(node_id=_NODE_A)]
    storage_handles = [MagicMock(), MagicMock()]
    server_info_refs = [handle.get_zmq_server_info.remote.return_value for handle in storage_handles]
    storage_unit = MagicMock()
    storage_unit.options.return_value.remote.side_effect = storage_handles
    timeout_error = ray.exceptions.GetTimeoutError()
    ready_at = {server_info_refs[0]: 40, server_info_refs[1]: 80}
    elapsed = 0
    kill = MagicMock(side_effect=[RuntimeError("Cleanup failed"), None] if cleanup_fails else None)

    # Advance readiness at the Ray boundary so scheduling delays cannot
    # change whether startup exceeds the shared timeout budget.
    def get(refs, timeout=None):
        nonlocal elapsed
        requested = refs if isinstance(refs, list) else [refs]
        wait = max(0, max(ready_at[ref] for ref in requested) - elapsed)
        if timeout is not None and wait > timeout:
            elapsed += timeout
            raise timeout_error
        elapsed += wait
        return [{} for _ in requested] if isinstance(refs, list) else {}

    monkeypatch.setattr(
        simple_storage_bootstrap, "get_node_round_robin_scheduling_strategies", lambda *_args, **_kwargs: strategies
    )
    monkeypatch.setattr(simple_storage_bootstrap, "SimpleStorageUnit", storage_unit)
    monkeypatch.setattr(ray, "available_resources", lambda: {"CPU": 1.0})
    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray, "kill", kill)

    conf = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {"num_data_storage_units": 2, "total_storage_size": None},
            }
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        simple_storage_bootstrap.initialize_simple_storage(conf)

    assert elapsed == 60
    assert "backend.SimpleStorage.num_data_storage_units=2" in str(exc_info.value)
    assert "requires Ray CPU capacity of 2" in str(exc_info.value)
    assert "available CPU capacity of 1" in str(exc_info.value)
    assert kill.call_args_list == [call(storage_handles[0]), call(storage_handles[1])]
    assert exc_info.value.__cause__ is timeout_error


def test_init_rolls_back_controller_when_storage_initialization_fails(monkeypatch):
    controller = MagicMock()
    controller_class = MagicMock()
    controller_class.options.return_value.remote.return_value = controller
    storage_error = RuntimeError("storage initialization failed")
    kill = MagicMock()

    monkeypatch.setattr(interface, "_TQ_CLIENT", None)
    monkeypatch.setattr(interface, "_TQ_STORAGE", None)
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", None)
    monkeypatch.setattr(interface, "_init_from_existing", lambda: False)
    monkeypatch.setattr(interface, "TransferQueueController", controller_class)
    monkeypatch.setattr(interface, "process_zmq_server_info", lambda _: {})
    monkeypatch.setattr(
        interface.StorageBootstrapProvider, "get_provider", lambda _: MagicMock(side_effect=storage_error)
    )
    monkeypatch.setattr(interface.ray, "kill", kill)

    with pytest.raises(RuntimeError) as exc_info:
        interface.init()

    assert exc_info.value is storage_error
    kill.assert_called_once_with(controller)
    assert interface._TQ_CONTROLLER is None
    assert interface._TQ_STORAGE is None


def test_init_can_reattach_after_controller_dies_while_waiting(monkeypatch):
    controller, replacement = MagicMock(), MagicMock()
    controller_error = ray.exceptions.ActorDiedError()
    conf = OmegaConf.create({})
    create_client = MagicMock()

    monkeypatch.setattr(interface, "_TQ_CONTROLLER", None)
    monkeypatch.setattr(interface, "_TQ_CLIENT", None)
    monkeypatch.setattr(interface.ray, "get_actor", MagicMock(side_effect=[controller, replacement]))
    monkeypatch.setattr(interface.ray, "get", MagicMock(side_effect=[None, controller_error, conf]))
    monkeypatch.setattr(interface.time, "sleep", lambda _: None)
    monkeypatch.setattr(interface, "_maybe_create_tq_client", create_client)

    with pytest.raises(ray.exceptions.ActorDiedError) as exc_info:
        interface.init()

    assert exc_info.value is controller_error
    assert interface._TQ_CONTROLLER is None
    create_client.assert_not_called()

    interface.init()

    assert interface._TQ_CONTROLLER is replacement
    create_client.assert_called_once_with(conf)


def test_controller_failure_does_not_reattach_an_existing_client(monkeypatch):
    controller, client = MagicMock(), MagicMock()
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", controller)
    monkeypatch.setattr(interface, "_TQ_CLIENT", client)
    monkeypatch.setattr(interface.ray, "get", MagicMock(side_effect=ray.exceptions.ActorDiedError()))

    with pytest.raises(ray.exceptions.ActorDiedError):
        interface.init()

    assert interface._TQ_CONTROLLER is controller
    assert interface._TQ_CLIENT is client


@pytest.mark.parametrize("failure_during_creation", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_simple_storage_cleans_up_partial_initialization(monkeypatch, failure_during_creation, cleanup_fails):
    storage_error = (
        ValueError("Actor name already exists")
        if failure_during_creation
        else ray.exceptions.ActorUnschedulableError("Storage node has no CPUs")
    )
    storage_handles = [MagicMock(), MagicMock()]
    storage_unit = MagicMock()
    storage_unit.options.return_value.remote.side_effect = (
        [storage_handles[0], storage_error] if failure_during_creation else storage_handles
    )
    created_handles = storage_handles[:1] if failure_during_creation else storage_handles
    kill = MagicMock(side_effect=[RuntimeError("Cleanup failed"), None] if cleanup_fails else None)

    monkeypatch.setattr(
        simple_storage_bootstrap,
        "get_node_round_robin_scheduling_strategies",
        lambda *_args, **_kwargs: [MagicMock(node_id=_NODE_A), MagicMock(node_id=_NODE_A)],
    )
    monkeypatch.setattr(simple_storage_bootstrap, "SimpleStorageUnit", storage_unit)
    monkeypatch.setattr(ray, "available_resources", lambda: {"CPU": 1.0})
    monkeypatch.setattr(ray, "get", MagicMock(side_effect=storage_error))
    monkeypatch.setattr(ray, "kill", kill)

    conf = OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {"num_data_storage_units": 2, "total_storage_size": None},
            }
        }
    )

    with pytest.raises(type(storage_error)) as exc_info:
        simple_storage_bootstrap.initialize_simple_storage(conf)

    assert exc_info.value is storage_error
    assert kill.call_args_list == [call(handle) for handle in created_handles]
