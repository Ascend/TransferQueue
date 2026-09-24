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

"""Publication, reads, and recovery coordinate across processes."""

import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from transfer_queue import data_dump, interface


def _configure():
    interface._TQ_CONTROLLER = object()
    interface._maybe_create_tq_client = lambda: SimpleNamespace(validate_dump_schema=lambda *_: None)


def _publisher(path, paused, release):
    _configure()
    rename = Path.rename

    def pause_at_publish(source, target):
        if source == path.with_name(path.name + ".tmp"):
            paused.set()
            if not release.wait(20):
                raise RuntimeError("Test publisher was not released")
        return rename(source, target)

    Path.rename = pause_at_publish
    data_dump.dump_data_by_key(path, [], "new")


def _reader(path, started, results):
    started.set()
    results.put(data_dump.read_row_index(path)["partition_id"])


def _loader(path, paused, release):
    _configure()

    def pause_load(*args):
        paused.set()
        if not release.wait(20):
            raise RuntimeError("Test loader was not released")

    data_dump._load_via_kv = pause_load
    data_dump.load_data_by_key(path)


@pytest.mark.parametrize("kill_writer", [False, True])
def test_reader_waits_for_publisher_and_recovers_after_exit(tmp_path, monkeypatch, kill_writer):
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", object())
    monkeypatch.setattr(interface, "_maybe_create_tq_client", lambda: object())
    path = tmp_path / "dump"
    data_dump.dump_data_by_key(path, [], "old")
    context = multiprocessing.get_context("spawn")
    paused, release, reader_started = context.Event(), context.Event(), context.Event()
    results = context.Queue()
    writer = context.Process(target=_publisher, args=(path, paused, release))
    reader = context.Process(target=_reader, args=(path, reader_started, results))
    writer.start()
    try:
        assert paused.wait(20)
        reader.start()
        assert reader_started.wait(20)
        reader.join(0.2)
        assert reader.is_alive(), "Reader rolled back a live publisher"
        if kill_writer:
            writer.terminate()
        else:
            release.set()
        writer.join(20)
        assert not writer.is_alive()
        assert results.get(timeout=20) == ("old" if kill_writer else "new")
        reader.join(20)
        assert reader.exitcode == 0
        if not kill_writer:
            assert writer.exitcode == 0
        assert path.with_name("dump.lock").exists()
    finally:
        # A terminated process may have held the Event's semaphore; do not reuse it.
        for process in (reader, writer):
            if process.pid and process.is_alive():
                process.terminate()
                process.join(10)
        results.close()


def test_publish_waits_until_load_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", object())
    monkeypatch.setattr(interface, "_maybe_create_tq_client", lambda: object())
    path = tmp_path / "dump"
    data_dump.dump_data_by_key(path, [], "old")
    context = multiprocessing.get_context("spawn")
    load_paused, load_release, publish_paused, publish_release = [context.Event() for _ in range(4)]
    loader = context.Process(target=_loader, args=(path, load_paused, load_release))
    writer = context.Process(target=_publisher, args=(path, publish_paused, publish_release))
    loader.start()
    try:
        assert load_paused.wait(20)
        writer.start()
        assert not publish_paused.wait(0.5)
        assert path.exists()
        load_release.set()
        loader.join(20)
        assert loader.exitcode == 0
        assert publish_paused.wait(20)
        publish_release.set()
        writer.join(20)
        assert writer.exitcode == 0
        assert data_dump.read_row_index(path)["partition_id"] == "new"
    finally:
        load_release.set()
        publish_release.set()
        for process in (writer, loader):
            if process.pid and process.is_alive():
                process.terminate()
                process.join(10)
