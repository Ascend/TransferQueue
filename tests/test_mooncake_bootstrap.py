# Copyright 2026 The TransferQueue Team
# SPDX-License-Identifier: Apache-2.0

"""A replacement master must wait for the previous listener to release its port."""

import socket
import subprocess
import sys
from pathlib import Path

import psutil
import pytest
from omegaconf import OmegaConf

from transfer_queue.storage.bootstrap import mooncake_bootstrap


@pytest.mark.parametrize("non_child", [False, True], ids=["own-child", "other-parent"])
@pytest.mark.parametrize("ignore_term", [False, True], ids=["delayed-exit", "forced-exit"])
def test_replacement_waits_for_old_listener(monkeypatch, ignore_term, non_child):
    if non_child and sys.platform != "linux":
        pytest.skip("Non-child zombie regression requires Linux")
    script = """
import signal, socket, sys, time
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
def stop(*args):
    time.sleep(0.3)
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if sys.argv[1] == "True" else stop)
print(listener.getsockname()[1], flush=True)
while True:
    time.sleep(1)
"""
    command = [sys.executable, "-u", "-c", script, str(ignore_term)]
    if non_child:
        owner_script = """
import subprocess, sys
master = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, text=True)
try:
    port = master.stdout.readline().strip()
    print(master.pid, port, flush=True)
    sys.stdin.readline()
finally:
    if master.poll() is None:
        master.kill()
    master.wait(timeout=5)
    master.stdout.close()
"""
        command = [sys.executable, "-u", "-c", owner_script, *command]
    old = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        if non_child:
            master_pid, port = map(int, old.stdout.readline().split())
        else:
            master_pid, port = old.pid, int(old.stdout.readline())
        master = psutil.Process(master_pid)
        monkeypatch.setattr(
            mooncake_bootstrap.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=f"{master_pid}\n"),
        )

        def launch_replacement(*args, **kwargs):
            if non_child:
                assert old.poll() is None, "The master owner must stay alive without reaping its child"
                assert master.status() == psutil.STATUS_ZOMBIE
            else:
                assert old.poll() is not None, "Replacement launched before the old master exited"
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", port))
            raise RuntimeError("Replacement boundary reached")

        monkeypatch.setattr(mooncake_bootstrap.subprocess, "Popen", launch_replacement)
        with pytest.raises(RuntimeError, match="Replacement boundary reached"):
            config = OmegaConf.load(Path(__file__).resolve().parents[1] / "transfer_queue" / "config.yaml")
            mooncake_bootstrap.initialize_mooncake_storage(config)
    finally:
        old.stdin.close()
        if not non_child and old.poll() is None:
            old.kill()
        old.wait(timeout=5)
        old.stdout.close()
