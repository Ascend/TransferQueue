#!/usr/bin/env python3
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
"""
MooncakeStore PUT/GET latency breakdown test.

Captures all [TQ-TIMING] print lines from every layer of the stack:
  interface.py   → kv_batch_put / kv_batch_get
  client.py      → client::put  / client::get_data
  base.py        → storage_mgr::get_data
  mooncake_client.py → mooncake::batch_put_tensor / mooncake::batch_get_tensor  (×N batches)

See CALLSTACK.md for the full call graph and measurement-point descriptions.

Breakdown available from Python:
  PUT: retrieve_meta | mooncake_batch_put (RDMA) | set_custom_meta
  GET: retrieve_meta | mooncake_batch_get (BatchQuery+RDMA, unsplit) | merge_to_tensordict
  Note: BatchQuery vs. RDMA split inside each batch_get_tensor requires C++ instrumentation.

Usage:
  python breakdown_test.py \\
    --backend_config perftest_config.yaml \\
    --head_node_ip 10.x.x.1 \\
    --worker_node_ip 10.x.x.2 \\
    [--device cpu] [--num_warmup 1] [--num_iters 3] \\
    [--scales small medium] [--output_dir ./results]
"""

import argparse
import csv
import io
import logging
import math
import os
import re
import sys
import time
from contextlib import redirect_stdout
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE_LIMIT = 500  # must match mooncake_client.py

# ── Scale definitions ──────────────────────────────────────────────────────────
# (label, batch_size, field_num, seq_len)
# Matches the "Small" / "Medium" settings in run_perf_test.sh
SCALE_PRESETS = {
    "small":  (1024,  9,  8192),
    "medium": (4096, 15, 32768),
    "large":  (8192, 18, 100000),
}


def data_size_gb(batch_size: int, field_num: int, seq_len: int) -> float:
    return batch_size * field_num * seq_len * 4 / 1024**3  # float32


def batchquery_calls(batch_size: int, field_num: int) -> int:
    return math.ceil(batch_size * field_num / BATCH_SIZE_LIMIT)


# ── [TQ-TIMING] log parser ─────────────────────────────────────────────────────
#
# Each captured stdout from an operation contains lines like:
#   [TQ-TIMING] kv_batch_put n=1024: retrieve_meta=0.0052s  storage_put=0.2358s  total=0.2410s
#   [TQ-TIMING] client::put n=1024: storage_put=0.2341s  set_custom_meta=0.0017s
#   [TQ-TIMING] storage_mgr::put_data n=1024 fields=9: key_val_gen=0.0012s  storage_put=0.2341s  meta_processing=0.0005s  notify=0.0010s
#   [TQ-TIMING] mooncake::batch_put_tensor n=500 data=0.0143GB  rdma=0.0245s  bw=4.66Gb/s
#   [TQ-TIMING] mooncake::put_loop n=9216 batches=19: rdma_total=0.0841s  py_overhead=0.0196s  loop_total=0.1037s
#   ... (one mooncake line per batch of BATCH_SIZE_LIMIT keys)
#
# For GET:
#   [TQ-TIMING] kv_batch_get n=1024: retrieve_meta=0.0048s  storage_get=1.5237s  total=1.5285s
#   [TQ-TIMING] client::get_data n=1024: storage_get=1.5237s
#   [TQ-TIMING] storage_mgr::get_data n=1024 fields=9: key_gen=0.0030s  rdma_get=1.5203s  merge_to_tensordict=0.0034s
#   [TQ-TIMING] mooncake::batch_get_tensor n=500 data=0.0143GB  rdma=0.0801s  bw=1.43Gb/s
#   [TQ-TIMING] mooncake::get_loop n=9216 batches=19: rdma_total=1.0000s  py_overhead=0.0580s  loop_total=1.0580s
#   ... (one mooncake line per batch of BATCH_SIZE_LIMIT keys)

_KV_FLOAT = r"([\d.]+)s"


def _f(name: str) -> str:
    """Regex fragment: key=value"""
    return rf"{name}={_KV_FLOAT}"


def parse_put_timings(captured: str) -> dict[str, float]:
    """Extract per-layer PUT timings from captured [TQ-TIMING] stdout."""
    result: dict[str, float] = {}

    # kv_batch_put line
    m = re.search(
        r"\[TQ-TIMING\] kv_batch_put[^:]*: " + _f("retrieve_meta") + r"\s+" + _f("storage_put") + r"\s+" + _f("total"),
        captured,
    )
    if m:
        result["kv_batch_put.retrieve_meta_s"] = float(m.group(1))
        result["kv_batch_put.storage_put_s"]   = float(m.group(2))
        result["kv_batch_put.total_s"]          = float(m.group(3))

    # client::put line
    m = re.search(
        r"\[TQ-TIMING\] client::put[^:]*: " + _f("storage_put") + r"\s+" + _f("set_custom_meta"),
        captured,
    )
    if m:
        result["client.put.storage_put_s"]      = float(m.group(1))
        result["client.put.set_custom_meta_s"]  = float(m.group(2))

    # storage_mgr::put_data line (NEW)
    m = re.search(
        r"\[TQ-TIMING\] storage_mgr::put_data[^:]*: "
        + _f("key_val_gen") + r"\s+" + _f("storage_put") + r"\s+"
        + _f("meta_processing") + r"\s+" + _f("notify"),
        captured,
    )
    if m:
        result["storage_mgr.put.key_val_gen_s"]      = float(m.group(1))
        result["storage_mgr.put.storage_put_s"]       = float(m.group(2))
        result["storage_mgr.put.meta_processing_s"]   = float(m.group(3))
        result["storage_mgr.put.notify_s"]            = float(m.group(4))

    # mooncake::batch_put_tensor lines (sum over all batches)
    mc_rdma_values = [float(v) for v in re.findall(
        r"\[TQ-TIMING\] mooncake::batch_put_tensor[^\n]*rdma=" + _KV_FLOAT,
        captured,
    )]
    result["mooncake.batch_put_tensor.count"]    = len(mc_rdma_values)
    result["mooncake.batch_put_tensor.total_s"]  = sum(mc_rdma_values)
    result["mooncake.batch_put_tensor.avg_s"]    = (
        sum(mc_rdma_values) / len(mc_rdma_values) if mc_rdma_values else 0.0
    )

    # mooncake::put_loop summary line
    m = re.search(
        r"\[TQ-TIMING\] mooncake::put_loop[^:]*: "
        + _f("rdma_total") + r"\s+" + _f("py_overhead") + r"\s+" + _f("loop_total"),
        captured,
    )
    if m:
        result["mooncake.put_loop.rdma_total_s"]  = float(m.group(1))
        result["mooncake.put_loop.py_overhead_s"]  = float(m.group(2))
        result["mooncake.put_loop.loop_total_s"]   = float(m.group(3))

    # mooncake::gdr_batch_put lines (GDR path — staging buffer)
    mc_gdr_d2d = [float(v) for v in re.findall(
        r"\[TQ-TIMING\] mooncake::gdr_batch_put[^\n]*d2d=" + _KV_FLOAT, captured)]
    mc_gdr_rdma = [float(v) for v in re.findall(
        r"\[TQ-TIMING\] mooncake::gdr_batch_put[^\n]*rdma=" + _KV_FLOAT, captured)]
    result["mooncake.gdr_batch_put.count"]      = len(mc_gdr_d2d)
    result["mooncake.gdr_batch_put.d2d_total_s"] = sum(mc_gdr_d2d)
    result["mooncake.gdr_batch_put.rdma_total_s"] = sum(mc_gdr_rdma)

    # mooncake::gdr_put_loop summary
    m = re.search(
        r"\[TQ-TIMING\] mooncake::gdr_put_loop[^:]*: "
        + _f("rdma_total") + r"\s+" + _f("d2d_total") + r"\s+" + _f("loop_total"),
        captured,
    )
    if m:
        result["mooncake.gdr_put_loop.rdma_total_s"] = float(m.group(1))
        result["mooncake.gdr_put_loop.d2d_total_s"]  = float(m.group(2))
        result["mooncake.gdr_put_loop.loop_total_s"]  = float(m.group(3))

    # mooncake::put (top-level client put with classify overhead)
    m = re.search(
        r"\[TQ-TIMING\] mooncake::put n=[^:]*: "
        + _f("classify") + r"\s+" + _f("total"),
        captured,
    )
    if m:
        result["mooncake.put.classify_s"] = float(m.group(1))
        result["mooncake.put.total_s"]    = float(m.group(2))

    return result


def parse_get_timings(captured: str) -> dict[str, float]:
    """Extract per-layer GET timings from captured [TQ-TIMING] stdout."""
    result: dict[str, float] = {}

    # kv_batch_get line
    m = re.search(
        r"\[TQ-TIMING\] kv_batch_get[^:]*: " + _f("retrieve_meta") + r"\s+" + _f("storage_get") + r"\s+" + _f("total"),
        captured,
    )
    if m:
        result["kv_batch_get.retrieve_meta_s"] = float(m.group(1))
        result["kv_batch_get.storage_get_s"]   = float(m.group(2))
        result["kv_batch_get.total_s"]          = float(m.group(3))

    # client::get_data line
    m = re.search(
        r"\[TQ-TIMING\] client::get_data[^:]*: " + _f("storage_get"),
        captured,
    )
    if m:
        result["client.get_data.storage_get_s"] = float(m.group(1))

    # storage_mgr::get_data line (UPDATED: now includes key_gen)
    m = re.search(
        r"\[TQ-TIMING\] storage_mgr::get_data[^:]*: "
        + _f("key_gen") + r"\s+" + _f("rdma_get") + r"\s+" + _f("merge_to_tensordict"),
        captured,
    )
    if m:
        result["storage_mgr.key_gen_s"]               = float(m.group(1))
        result["storage_mgr.rdma_get_s"]               = float(m.group(2))
        result["storage_mgr.merge_to_tensordict_s"]    = float(m.group(3))
    else:
        # Fallback: old format without key_gen
        m = re.search(
            r"\[TQ-TIMING\] storage_mgr::get_data[^:]*: " + _f("rdma_get") + r"\s+" + _f("merge_to_tensordict"),
            captured,
        )
        if m:
            result["storage_mgr.key_gen_s"]               = 0.0
            result["storage_mgr.rdma_get_s"]               = float(m.group(1))
            result["storage_mgr.merge_to_tensordict_s"]    = float(m.group(2))

    # mooncake::batch_get_tensor lines (sum over all batches)
    # NOTE: each batch's rdma= includes BOTH BatchQuery (gRPC) + RDMA read.
    # The split requires C++ instrumentation (see CALLSTACK.md).
    mc_rdma_values = [float(v) for v in re.findall(
        r"\[TQ-TIMING\] mooncake::batch_get_tensor[^\n]*rdma=" + _KV_FLOAT,
        captured,
    )]
    result["mooncake.batch_get_tensor.count"]    = len(mc_rdma_values)
    result["mooncake.batch_get_tensor.total_s"]  = sum(mc_rdma_values)
    result["mooncake.batch_get_tensor.avg_s"]    = (
        sum(mc_rdma_values) / len(mc_rdma_values) if mc_rdma_values else 0.0
    )

    # mooncake::get_loop summary line
    m = re.search(
        r"\[TQ-TIMING\] mooncake::get_loop[^:]*: "
        + _f("rdma_total") + r"\s+" + _f("py_overhead") + r"\s+" + _f("loop_total"),
        captured,
    )
    if m:
        result["mooncake.get_loop.rdma_total_s"]  = float(m.group(1))
        result["mooncake.get_loop.py_overhead_s"]  = float(m.group(2))
        result["mooncake.get_loop.loop_total_s"]   = float(m.group(3))

    # mooncake::gdr_batch_get lines (GDR path — staging buffer)
    mc_gdr_rdma = [float(v) for v in re.findall(
        r"\[TQ-TIMING\] mooncake::gdr_batch_get[^\n]*rdma=" + _KV_FLOAT, captured)]
    mc_gdr_d2d = [float(v) for v in re.findall(
        r"\[TQ-TIMING\] mooncake::gdr_batch_get[^\n]*d2d=" + _KV_FLOAT, captured)]
    result["mooncake.gdr_batch_get.count"]       = len(mc_gdr_rdma)
    result["mooncake.gdr_batch_get.rdma_total_s"] = sum(mc_gdr_rdma)
    result["mooncake.gdr_batch_get.d2d_total_s"]  = sum(mc_gdr_d2d)

    # mooncake::gdr_get_loop summary
    m = re.search(
        r"\[TQ-TIMING\] mooncake::gdr_get_loop[^:]*: "
        + _f("rdma_total") + r"\s+" + _f("d2d_total") + r"\s+" + _f("loop_total"),
        captured,
    )
    if m:
        result["mooncake.gdr_get_loop.rdma_total_s"] = float(m.group(1))
        result["mooncake.gdr_get_loop.d2d_total_s"]  = float(m.group(2))
        result["mooncake.gdr_get_loop.loop_total_s"]  = float(m.group(3))

    # mooncake::get (top-level client get with classify/scatter/gather)
    m = re.search(
        r"\[TQ-TIMING\] mooncake::get n=[^:]*: "
        + _f("classify") + r"\s+" + _f("scatter") + r"\s+"
        + _f("gather") + r"\s+" + _f("total"),
        captured,
    )
    if m:
        result["mooncake.get.classify_s"] = float(m.group(1))
        result["mooncake.get.scatter_s"]  = float(m.group(2))
        result["mooncake.get.gather_s"]   = float(m.group(3))
        result["mooncake.get.total_s"]    = float(m.group(4))

    return result


# ── Ray actor ─────────────────────────────────────────────────────────────────

@ray.remote
class BreakdownActor:
    """Ray actor that runs TQ PUT/GET and captures all [TQ-TIMING] stdout lines."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.test_keys: list[str] = []
        self.test_data: TensorDict | None = None

    def initialize(self) -> None:
        tq.init(OmegaConf.create(self.config))

    def prepare_data(self, batch_size: int, field_num: int, seq_len: int, device: str = "cpu") -> float:
        self.test_keys = [f"sample_{i}" for i in range(batch_size)]
        td = TensorDict(batch_size=(batch_size,))
        for f in range(field_num):
            t = torch.randn(batch_size, seq_len, dtype=torch.float32)
            if device == "gpu":
                t = t.cuda()
            td.set(f"field_{f}", t)
        self.test_data = td
        return data_size_gb(batch_size, field_num, seq_len)

    def put_timed(self, partition_id: str) -> tuple[float, str]:
        """
        Run kv_batch_put. Returns (wall_clock_s, captured_tq_timing_stdout).

        Captures all [TQ-TIMING] prints from:
          interface.py  → kv_batch_put
          client.py     → client::put
          mooncake_client.py → mooncake::batch_put_tensor (×N batches)
        """
        buf = io.StringIO()
        t0 = time.perf_counter()
        with redirect_stdout(buf):
            tq.kv_batch_put(keys=self.test_keys, partition_id=partition_id, fields=self.test_data)
        wall = time.perf_counter() - t0
        return wall, buf.getvalue()

    def list_keys(self, partition_id: str) -> list[str]:
        info = tq.kv_list(partition_id=partition_id)
        return list(info.get(partition_id, {}).keys())

    def get_timed(self, partition_id: str, keys: list[str]) -> tuple[float, str]:
        """
        Run kv_batch_get. Returns (wall_clock_s, captured_tq_timing_stdout).

        Captures all [TQ-TIMING] prints from:
          interface.py  → kv_batch_get
          client.py     → client::get_data
          base.py       → storage_mgr::get_data
          mooncake_client.py → mooncake::batch_get_tensor (×N batches)
        """
        buf = io.StringIO()
        t0 = time.perf_counter()
        with redirect_stdout(buf):
            tq.kv_batch_get(keys=keys, partition_id=partition_id)
        wall = time.perf_counter() - t0
        return wall, buf.getvalue()

    def delete(self, partition_id: str, keys: list[str]) -> None:
        tq.kv_clear(keys=keys, partition_id=partition_id)

    def close(self) -> None:
        tq.close()


# ── Config & actor setup ───────────────────────────────────────────────────────

def prepare_config(
    backend_config_path: str, head_node_ip: str, worker_node_ip: str | None
) -> dict[str, Any]:
    config = OmegaConf.load(backend_config_path)
    assert str(config.backend.storage_backend) == "MooncakeStore", (
        "breakdown_test.py is for MooncakeStore only (got "
        f"{config.backend.storage_backend})"
    )
    if worker_node_ip is not None:
        mc = config.backend.MooncakeStore
        for field in ("metadata_server", "master_server_address"):
            val = str(getattr(mc, field, "") or "")
            for placeholder in ("localhost", "127.0.0.1"):
                if val.startswith(placeholder):
                    new_val = val.replace(placeholder, head_node_ip, 1)
                    setattr(mc, field, new_val)
                    logger.info(f"Inter-node: override {field}: {val} → {new_val}")
    return OmegaConf.to_container(config, resolve=True)


def create_actors(
    config: dict[str, Any],
    head_node_ip: str,
    worker_node_ip: str | None,
    device: str,
) -> tuple[Any, Any]:
    writer_node = head_node_ip
    reader_node = worker_node_ip or head_node_ip

    def _opts(node_ip: str) -> dict:
        opts: dict[str, Any] = {
            "num_cpus": 0.001,
            "resources": {f"node:{node_ip}": 0.001},
        }
        if device == "gpu":
            opts["num_gpus"] = 1
        elif device == "npu":
            opts["resources"]["NPU"] = 1
        return opts

    writer = BreakdownActor.options(**_opts(writer_node)).remote(config)
    reader = BreakdownActor.options(**_opts(reader_node)).remote(config)
    ray.get([writer.initialize.remote(), reader.initialize.remote()])
    logger.info(f"Writer on {writer_node}, reader on {reader_node}")
    return writer, reader


# ── Single-iteration benchmark ─────────────────────────────────────────────────

def run_iteration(
    label: str,
    batch_size: int,
    field_num: int,
    writer: Any,
    reader: Any,
) -> dict[str, Any]:
    """Run one PUT+GET cycle; return flat timing dict."""
    partition_id = f"bkd_{label}"

    # PUT
    put_wall, put_captured = ray.get(writer.put_timed.remote(partition_id))
    time.sleep(1)

    # LIST keys (reader needs actual key list)
    keys = ray.get(reader.list_keys.remote(partition_id))

    # GET
    get_wall, get_captured = ray.get(reader.get_timed.remote(partition_id, keys))
    time.sleep(1)

    # DELETE
    ray.get(writer.delete.remote(partition_id, keys))
    time.sleep(1)

    # Parse TQ-TIMING logs
    put_t = parse_put_timings(put_captured)
    get_t = parse_get_timings(get_captured)

    return {
        "scale":       label,
        "batch_size":  batch_size,
        "field_num":   field_num,
        "data_gb":     data_size_gb(batch_size, field_num, SCALE_PRESETS[label][2]),
        "bq_calls":    batchquery_calls(batch_size, field_num),
        # Wall-clock from caller (ray.get overhead included)
        "put_wall_s":  put_wall,
        "get_wall_s":  get_wall,
        # PUT breakdown from [TQ-TIMING] prints
        **{f"put.{k}": v for k, v in put_t.items()},
        # GET breakdown from [TQ-TIMING] prints
        **{f"get.{k}": v for k, v in get_t.items()},
        # Captured raw lines (for debugging)
        "_put_raw": put_captured,
        "_get_raw": get_captured,
    }


# ── Pretty-print summary ───────────────────────────────────────────────────────

def print_breakdown(r: dict[str, Any], label: str) -> None:
    s = label
    gb    = r["data_gb"]
    bq    = r["bq_calls"]

    # PUT - interface level
    p_rmeta   = r.get("put.kv_batch_put.retrieve_meta_s", 0) * 1e3
    p_scmeta  = r.get("put.client.put.set_custom_meta_s",  0) * 1e3
    p_total   = r.get("put.kv_batch_put.total_s",          0) * 1e3
    p_bw      = gb * 8 / (p_total / 1e3) if p_total else 0
    # PUT - storage_mgr level
    p_kvgen   = r.get("put.storage_mgr.put.key_val_gen_s",    0) * 1e3
    p_metaproc = r.get("put.storage_mgr.put.meta_processing_s", 0) * 1e3
    p_notify  = r.get("put.storage_mgr.put.notify_s",          0) * 1e3
    # PUT - mooncake client level
    p_classify = r.get("put.mooncake.put.classify_s", 0) * 1e3
    # PUT - detect GDR vs baseline path
    is_gdr = r.get("put.mooncake.gdr_batch_put.count", 0) > 0
    if is_gdr:
        pm_count  = int(r.get("put.mooncake.gdr_batch_put.count", 0))
        pm_d2d    = r.get("put.mooncake.gdr_put_loop.d2d_total_s",  0) * 1e3
        pm_rdma   = r.get("put.mooncake.gdr_put_loop.rdma_total_s", 0) * 1e3
        pm_loop   = r.get("put.mooncake.gdr_put_loop.loop_total_s", 0) * 1e3
        pm_pyoh   = max(0, pm_loop - pm_d2d - pm_rdma)
    else:
        pm_count  = int(r.get("put.mooncake.batch_put_tensor.count", 0))
        pm_d2d    = 0.0
        pm_rdma   = r.get("put.mooncake.put_loop.rdma_total_s",  0) * 1e3
        pm_pyoh   = r.get("put.mooncake.put_loop.py_overhead_s", 0) * 1e3
        pm_loop   = r.get("put.mooncake.put_loop.loop_total_s",  0) * 1e3
    # PUT - remaining gap
    p_accounted = p_rmeta + p_kvgen + p_classify + pm_loop + p_metaproc + p_notify + p_scmeta
    p_gap = max(0, p_total - p_accounted)

    # GET - interface level
    g_rmeta   = r.get("get.kv_batch_get.retrieve_meta_s", 0) * 1e3
    g_total   = r.get("get.kv_batch_get.total_s",          0) * 1e3
    g_bw      = gb * 8 / (g_total / 1e3) if g_total else 0
    # GET - storage_mgr level
    g_keygen  = r.get("get.storage_mgr.key_gen_s",              0) * 1e3
    g_merge   = r.get("get.storage_mgr.merge_to_tensordict_s",  0) * 1e3
    # GET - mooncake client level
    g_classify = r.get("get.mooncake.get.classify_s", 0) * 1e3
    g_scatter  = r.get("get.mooncake.get.scatter_s",  0) * 1e3
    g_gather   = r.get("get.mooncake.get.gather_s",   0) * 1e3
    # GET - detect GDR vs baseline path
    is_gdr_get = r.get("get.mooncake.gdr_batch_get.count", 0) > 0
    if is_gdr_get:
        gm_count  = int(r.get("get.mooncake.gdr_batch_get.count", 0))
        gm_d2d    = r.get("get.mooncake.gdr_get_loop.d2d_total_s",  0) * 1e3
        gm_rdma   = r.get("get.mooncake.gdr_get_loop.rdma_total_s", 0) * 1e3
        gm_loop   = r.get("get.mooncake.gdr_get_loop.loop_total_s", 0) * 1e3
        gm_pyoh   = max(0, gm_loop - gm_d2d - gm_rdma)
    else:
        gm_count  = int(r.get("get.mooncake.batch_get_tensor.count", 0))
        gm_d2d    = 0.0
        gm_rdma   = r.get("get.mooncake.get_loop.rdma_total_s",  0) * 1e3
        gm_pyoh   = r.get("get.mooncake.get_loop.py_overhead_s", 0) * 1e3
        gm_loop   = r.get("get.mooncake.get_loop.loop_total_s",  0) * 1e3
    # GET - remaining gap
    g_accounted = g_rmeta + g_keygen + g_classify + g_scatter + gm_loop + g_gather + g_merge
    g_gap = max(0, g_total - g_accounted)

    mode_tag = " [GDR]" if is_gdr else " [Baseline]"
    print(f"\n  ┌─ {s}{mode_tag}  ({gb:.3f} GB, {bq} BatchQuery calls/GET) {'─'*30}")
    print(f"  │  PUT  total={p_total:7.1f}ms  bw={p_bw:6.1f} Gb/s")
    print(f"  │    ├ retrieve_meta (ZMQ→controller)  : {p_rmeta:7.1f} ms")
    print(f"  │    ├ key_val_gen (Python)             : {p_kvgen:7.1f} ms")
    classify_desc = "classify (no D2H, GPU stays)" if is_gdr else "classify (D2H: .cuda()→.cpu())"
    print(f"  │    ├ {classify_desc:<37}: {p_classify:7.1f} ms")
    print(f"  │    ├ mooncake_loop ×{pm_count:<3}              : {pm_loop:7.1f} ms")
    if is_gdr:
        print(f"  │    │   ├ D2D copy (GPU→staging buf)  : {pm_d2d:7.1f} ms")
        print(f"  │    │   ├ GDR RDMA transfer            : {pm_rdma:7.1f} ms")
        print(f"  │    │   └ Python overhead              : {pm_pyoh:7.1f} ms")
    else:
        print(f"  │    │   ├ C++ batch_put_tensor         : {pm_rdma:7.1f} ms")
        print(f"  │    │   └ Python overhead (slice+valid): {pm_pyoh:7.1f} ms")
    print(f"  │    ├ meta_processing (Python)         : {p_metaproc:7.1f} ms")
    print(f"  │    ├ notify (ZMQ→controller)          : {p_notify:7.1f} ms")
    print(f"  │    ├ set_custom_meta (ZMQ→controller) : {p_scmeta:7.1f} ms")
    if p_gap > 0.5:
        print(f"  │    └ remaining gap                   : {p_gap:7.1f} ms")
    print(f"  │")
    print(f"  │  GET  total={g_total:7.1f}ms  bw={g_bw:6.1f} Gb/s")
    print(f"  │    ├ retrieve_meta (ZMQ→controller)  : {g_rmeta:7.1f} ms")
    print(f"  │    ├ key_gen (Python)                : {g_keygen:7.1f} ms")
    print(f"  │    ├ classify+scatter (Python)       : {g_classify + g_scatter:7.1f} ms")
    print(f"  │    ├ mooncake_loop ×{gm_count:<3}              : {gm_loop:7.1f} ms")
    if is_gdr_get:
        print(f"  │    │   ├ GDR RDMA transfer            : {gm_rdma:7.1f} ms")
        print(f"  │    │   ├ D2D copy (staging buf→GPU)  : {gm_d2d:7.1f} ms")
        print(f"  │    │   └ Python overhead              : {gm_pyoh:7.1f} ms")
    else:
        print(f"  │    │   ├ C++ batch_get_tensor         : {gm_rdma:7.1f} ms")
        print(f"  │    │   └ Python overhead (slice+valid): {gm_pyoh:7.1f} ms")
    print(f"  │    ├ gather (Python, result scatter)  : {g_gather:7.1f} ms")
    print(f"  │    ├ merge_to_tensordict (CPU)        : {g_merge:7.1f} ms")
    if g_gap > 0.5:
        print(f"  │    └ remaining gap                   : {g_gap:7.1f} ms")
    print(f"  └{'─'*60}")


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(results: list[dict[str, Any]], path: str) -> None:
    # Exclude raw captured stdout from CSV (too large)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Results saved → {path}")


def save_raw_log(results: list[dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(f"=== scale={r['scale']} iter={r.get('iter','')} ===\n")
            f.write("--- PUT captured stdout ---\n")
            f.write(r.get("_put_raw", "") + "\n")
            f.write("--- GET captured stdout ---\n")
            f.write(r.get("_get_raw", "") + "\n")
    logger.info(f"Raw [TQ-TIMING] log saved → {path}")


# ── Chart ──────────────────────────────────────────────────────────────────────

def draw_chart(results: list[dict[str, Any]], out_path: str) -> None:
    """
    Two-panel stacked bar chart with full breakdown including Python overhead.
      Left panel:  PUT breakdown
      Right panel: GET breakdown
    Each bar's components sum to total; unmeasured gap shown as hatched red.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed; skipping chart generation.")
        return

    from collections import defaultdict
    import statistics

    # Aggregate across iterations per scale
    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        sc = r["scale"]
        agg[sc]["data_gb"].append(r["data_gb"])
        agg[sc]["bq_calls"].append(r["bq_calls"])

        p_total = r.get("put.kv_batch_put.total_s", 0) * 1e3
        p_rmeta = r.get("put.kv_batch_put.retrieve_meta_s", 0) * 1e3
        p_kvgen = r.get("put.storage_mgr.put.key_val_gen_s", 0) * 1e3
        p_classify = r.get("put.mooncake.put.classify_s", 0) * 1e3
        # Detect GDR vs baseline for PUT
        if r.get("put.mooncake.gdr_batch_put.count", 0) > 0:
            p_d2d   = r.get("put.mooncake.gdr_put_loop.d2d_total_s", 0) * 1e3
            p_rdma  = r.get("put.mooncake.gdr_put_loop.rdma_total_s", 0) * 1e3
            p_loop  = r.get("put.mooncake.gdr_put_loop.loop_total_s", 0) * 1e3
            p_pyoh  = max(0, p_loop - p_d2d - p_rdma)
        else:
            p_d2d   = 0.0
            p_rdma  = r.get("put.mooncake.put_loop.rdma_total_s", 0) * 1e3
            p_pyoh  = r.get("put.mooncake.put_loop.py_overhead_s", 0) * 1e3
        p_metap = r.get("put.storage_mgr.put.meta_processing_s", 0) * 1e3
        p_notify = r.get("put.storage_mgr.put.notify_s", 0) * 1e3
        p_scm   = r.get("put.client.put.set_custom_meta_s", 0) * 1e3
        p_gap   = max(0, p_total - p_rmeta - p_kvgen - p_classify - p_d2d - p_rdma - p_pyoh - p_metap - p_notify - p_scm)

        agg[sc]["put.total"].append(p_total)
        agg[sc]["put.retrieve_meta"].append(p_rmeta)
        agg[sc]["put.key_val_gen"].append(p_kvgen)
        agg[sc]["put.classify"].append(p_classify)
        agg[sc]["put.d2d"].append(p_d2d)
        agg[sc]["put.rdma"].append(p_rdma)
        agg[sc]["put.loop_pyoh"].append(p_pyoh)
        agg[sc]["put.meta_processing"].append(p_metap)
        agg[sc]["put.notify"].append(p_notify)
        agg[sc]["put.set_custom_meta"].append(p_scm)
        agg[sc]["put.gap"].append(p_gap)

        g_total = r.get("get.kv_batch_get.total_s", 0) * 1e3
        g_rmeta = r.get("get.kv_batch_get.retrieve_meta_s", 0) * 1e3
        g_keygen = r.get("get.storage_mgr.key_gen_s", 0) * 1e3
        g_classify = r.get("get.mooncake.get.classify_s", 0) * 1e3
        g_scatter = r.get("get.mooncake.get.scatter_s", 0) * 1e3
        # Detect GDR vs baseline for GET
        if r.get("get.mooncake.gdr_batch_get.count", 0) > 0:
            g_d2d   = r.get("get.mooncake.gdr_get_loop.d2d_total_s", 0) * 1e3
            g_rdma  = r.get("get.mooncake.gdr_get_loop.rdma_total_s", 0) * 1e3
            g_loop  = r.get("get.mooncake.gdr_get_loop.loop_total_s", 0) * 1e3
            g_pyoh  = max(0, g_loop - g_d2d - g_rdma)
        else:
            g_d2d   = 0.0
            g_rdma  = r.get("get.mooncake.get_loop.rdma_total_s", 0) * 1e3
            g_pyoh  = r.get("get.mooncake.get_loop.py_overhead_s", 0) * 1e3
        g_gather = r.get("get.mooncake.get.gather_s", 0) * 1e3
        g_merge = r.get("get.storage_mgr.merge_to_tensordict_s", 0) * 1e3
        g_gap   = max(0, g_total - g_rmeta - g_keygen - g_classify - g_scatter - g_d2d - g_rdma - g_pyoh - g_gather - g_merge)

        agg[sc]["get.total"].append(g_total)
        agg[sc]["get.retrieve_meta"].append(g_rmeta)
        agg[sc]["get.key_gen"].append(g_keygen)
        agg[sc]["get.classify_scatter"].append(g_classify + g_scatter)
        agg[sc]["get.d2d"].append(g_d2d)
        agg[sc]["get.rdma"].append(g_rdma)
        agg[sc]["get.loop_pyoh"].append(g_pyoh)
        agg[sc]["get.gather"].append(g_gather)
        agg[sc]["get.merge"].append(g_merge)
        agg[sc]["get.gap"].append(g_gap)

    scale_order = [s for s in ("small", "medium", "large") if s in agg]
    n = len(scale_order)
    if n == 0:
        return

    def mean(lst):
        return statistics.mean(lst) if lst else 0.0

    gb_arr = np.array([mean(agg[s]["data_gb"]) for s in scale_order])
    bq_arr = [int(mean(agg[s]["bq_calls"])) for s in scale_order]

    # PUT arrays
    put_rmeta  = np.array([mean(agg[s]["put.retrieve_meta"]) for s in scale_order])
    put_kvgen  = np.array([mean(agg[s]["put.key_val_gen"]) for s in scale_order])
    put_classify = np.array([mean(agg[s]["put.classify"]) for s in scale_order])
    put_d2d    = np.array([mean(agg[s]["put.d2d"]) for s in scale_order])
    put_rdma   = np.array([mean(agg[s]["put.rdma"]) for s in scale_order])
    put_pyoh   = np.array([mean(agg[s]["put.loop_pyoh"]) for s in scale_order])
    put_metap  = np.array([mean(agg[s]["put.meta_processing"]) for s in scale_order])
    put_notify = np.array([mean(agg[s]["put.notify"]) for s in scale_order])
    put_scm    = np.array([mean(agg[s]["put.set_custom_meta"]) for s in scale_order])
    put_gap    = np.array([mean(agg[s]["put.gap"]) for s in scale_order])
    put_total  = np.array([mean(agg[s]["put.total"]) for s in scale_order])

    # GET arrays
    get_rmeta  = np.array([mean(agg[s]["get.retrieve_meta"]) for s in scale_order])
    get_keygen = np.array([mean(agg[s]["get.key_gen"]) for s in scale_order])
    get_d2d    = np.array([mean(agg[s]["get.d2d"]) for s in scale_order])
    get_rdma   = np.array([mean(agg[s]["get.rdma"]) for s in scale_order])
    get_pyoh   = np.array([mean(agg[s]["get.loop_pyoh"]) for s in scale_order])
    get_merge  = np.array([mean(agg[s]["get.merge"]) for s in scale_order])
    get_gap    = np.array([mean(agg[s]["get.gap"]) for s in scale_order])
    get_total  = np.array([mean(agg[s]["get.total"]) for s in scale_order])

    put_bw = gb_arr * 8 / (put_total / 1e3 + 1e-9)
    get_bw = gb_arr * 8 / (get_total / 1e3 + 1e-9)

    x = np.arange(n)
    bw = 0.55

    # Colors
    c_zmq    = "#5B9BD5"   # blue   - ZMQ calls
    c_keygen = "#A855F7"   # purple - key/value generation
    c_class  = "#E97451"   # coral  - classify (includes D2H for baseline)
    c_d2d    = "#17BECF"   # cyan   - D2D GPU copy (GDR only)
    c_rdma_p = "#70AD47"   # green  - RDMA put
    c_rdma_g = "#FF6B35"   # orange - RDMA get (gRPC+RDMA)
    c_pyoh   = "#D9534F"   # red    - Python loop overhead
    c_meta   = "#9DC3E6"   # light blue - meta processing
    c_merge  = "#FFD966"   # yellow - merge_to_tensordict
    c_gap    = "#888888"   # gray   - remaining gap

    has_d2d = put_d2d.max() > 0 or get_d2d.max() > 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ────── PUT panel ──────
    ax = axes[0]
    bottom = np.zeros(n)

    def _bar(ax, vals, label, color, hatch=None):
        nonlocal bottom
        ax.bar(x, vals, bw, bottom=bottom, label=label, color=color,
               zorder=3, edgecolor="white", linewidth=0.5, hatch=hatch)
        bottom += vals

    _bar(ax, put_rmeta,    "retrieve_meta (ZMQ)",                c_zmq)
    _bar(ax, put_kvgen,    "key_val_gen (Python)",                c_keygen)
    classify_label = "classify (no D2H)" if has_d2d else "classify (D2H: cuda→cpu)"
    if put_classify.max() > 0.5:
        _bar(ax, put_classify, classify_label,                    c_class)
    if put_d2d.max() > 0:
        _bar(ax, put_d2d,  "D2D copy (GPU→staging)",             c_d2d)
    _bar(ax, put_rdma,     "RDMA transfer" if has_d2d else "C++ batch_put_tensor", c_rdma_p)
    _bar(ax, put_pyoh,     "loop overhead (Python)",              c_pyoh, hatch="//")
    _bar(ax, put_metap,    "meta_processing (Python)",            c_meta)
    _bar(ax, put_notify,   "notify (ZMQ)",                        c_zmq)
    _bar(ax, put_scm,      "set_custom_meta (ZMQ)",               c_zmq)
    if put_gap.max() > 0.5:
        _bar(ax, put_gap,  "remaining gap",                       c_gap, hatch="xx")

    for i in range(n):
        ax.text(i, put_total[i] + max(put_total) * 0.02,
                f"{put_total[i]:.0f}ms\n{put_bw[i]:.1f} Gb/s",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_title("PUT Breakdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{gb_arr[i]:.2f} GB\n{bq_arr[i]} batches"
                        for i, s in enumerate(scale_order)])
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=7.5, loc="upper left")

    # ────── GET panel ──────
    ax = axes[1]
    bottom = np.zeros(n)

    _bar(ax, get_rmeta,  "retrieve_meta (ZMQ)",        c_zmq)
    _bar(ax, get_keygen, "key_gen (Python)",            c_keygen)
    _bar(ax, get_rdma,   "RDMA transfer" if has_d2d else "C++ batch_get_tensor", c_rdma_g)
    if get_d2d.max() > 0:
        _bar(ax, get_d2d, "D2D copy (staging→GPU)",    c_d2d)
    _bar(ax, get_pyoh,   "loop overhead (Python)",      c_pyoh, hatch="//")
    _bar(ax, get_merge,  "merge_to_tensordict (CPU)",   c_merge)
    if get_gap.max() > 0.5:
        _bar(ax, get_gap, "remaining gap",              c_gap, hatch="xx")

    for i in range(n):
        ax.text(i, get_total[i] + max(get_total) * 0.02,
                f"{get_total[i]:.0f}ms\n{get_bw[i]:.1f} Gb/s",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_title("GET Breakdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{gb_arr[i]:.2f} GB\n{bq_arr[i]} batches"
                        for i, s in enumerate(scale_order)])
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=7.5, loc="upper left")

    mode_label = "GDR (GPU Direct RDMA)" if has_d2d else "Baseline (CPU memcpy)"
    fig.suptitle(
        f"MooncakeStore Latency Breakdown — {mode_label}\n"
        "Green/Orange = RDMA transfer  |  Cyan = D2D GPU copy  |  Red hatched = Python overhead",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MooncakeStore PUT/GET latency breakdown test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend_config",  required=True,  help="Path to backend config YAML (must use MooncakeStore)")
    parser.add_argument("--head_node_ip",    required=True,  help="Head node IP (mooncake_master host, writer node)")
    parser.add_argument("--worker_node_ip",  default=None,   help="Reader node IP (omit for single-node test)")
    parser.add_argument("--device",          default="cpu",  choices=["cpu", "gpu", "npu"])
    parser.add_argument("--num_warmup",      type=int, default=1,  help="Warmup iterations per scale")
    parser.add_argument("--num_iters",       type=int, default=3,  help="Measured iterations per scale")
    parser.add_argument("--scales",          nargs="+", choices=list(SCALE_PRESETS), default=None,
                        help="Which scales to test (default: all)")
    parser.add_argument("--output_dir",      default="./results", help="Output directory for CSV and chart")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = prepare_config(args.backend_config, args.head_node_ip, args.worker_node_ip)

    ray.init(address="auto", ignore_reinit_error=True)
    writer, reader = create_actors(config, args.head_node_ip, args.worker_node_ip, args.device)

    scales_to_run = args.scales or list(SCALE_PRESETS.keys())
    all_results: list[dict[str, Any]] = []

    try:
        for label in scales_to_run:
            batch_size, field_num, seq_len = SCALE_PRESETS[label]
            gb = data_size_gb(batch_size, field_num, seq_len)
            bq = batchquery_calls(batch_size, field_num)

            logger.info(f"\n{'='*65}")
            logger.info(f"Scale: {label}  batch={batch_size}  fields={field_num}  seq={seq_len}")
            logger.info(f"  data={gb:.3f} GB  total_keys={batch_size*field_num}  BatchQuery_calls/GET={bq}")

            # Prepare test data on writer node
            ray.get(writer.prepare_data.remote(batch_size, field_num, seq_len, args.device))

            total_iters = args.num_warmup + args.num_iters
            for i in range(total_iters):
                is_warmup = i < args.num_warmup
                tag = f"warmup-{i+1}" if is_warmup else f"iter-{i - args.num_warmup + 1}/{args.num_iters}"
                logger.info(f"  [{label}] {tag}")

                r = run_iteration(label, batch_size, field_num, writer, reader)
                r["iter"] = -1 if is_warmup else (i - args.num_warmup + 1)

                # Always print breakdown (useful for warmup too)
                print_breakdown(r, f"{label} [{tag}]")

                if not is_warmup:
                    all_results.append(r)

            time.sleep(5)

    finally:
        ray.get([writer.close.remote(), reader.close.remote()])

    if not all_results:
        logger.error("No measured results collected.")
        sys.exit(1)

    # ── Summary table ──
    from collections import defaultdict
    import statistics

    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        sc = r["scale"]
        for k in r:
            if not k.startswith("_") and isinstance(r[k], float | int):
                agg[sc][k].append(r[k])

    print(f"\n{'='*95}")
    print(f"{'Scale':<8} {'GB':>6} {'BQ':>5} | {'PUT ms':>8} {'PUT Gb/s':>9} | "
          f"{'GET ms':>8} {'GET Gb/s':>9} | {'BQ overhead*':>13}")
    print(f"{'-'*95}")
    for sc in scales_to_run:
        if sc not in agg:
            continue
        d = agg[sc]
        put_ms = statistics.mean(d.get("put.kv_batch_put.total_s", [0])) * 1e3
        get_ms = statistics.mean(d.get("get.kv_batch_get.total_s", [0])) * 1e3
        gb_v   = statistics.mean(d.get("data_gb", [0]))
        bq_v   = int(statistics.mean(d.get("bq_calls", [0])))
        put_bw = gb_v * 8 / (put_ms / 1e3 + 1e-9)
        get_bw = gb_v * 8 / (get_ms / 1e3 + 1e-9)
        bq_est = get_ms - put_ms  # rough: all non-RDMA GET overhead
        print(f"{sc:<8} {gb_v:>6.3f} {bq_v:>5} | "
              f"{put_ms:>8.1f} {put_bw:>9.1f} | "
              f"{get_ms:>8.1f} {get_bw:>9.1f} | "
              f"{bq_est:>11.1f}ms")
    print(f"{'='*95}")
    print("* BQ overhead = GET_total - PUT_total (rough estimate; includes all non-RDMA GET latency)")
    print("  True BatchQuery time needs C++ instrumentation in real_client.cpp (see CALLSTACK.md)")

    # ── Save outputs ──
    csv_path   = os.path.join(args.output_dir, "breakdown_results.csv")
    log_path   = os.path.join(args.output_dir, "breakdown_raw_tqtiming.log")
    chart_path = os.path.join(args.output_dir, "breakdown_chart.png")

    save_csv(all_results, csv_path)
    save_raw_log(all_results, log_path)
    draw_chart(all_results, chart_path)

    logger.info("Breakdown test complete.")


if __name__ == "__main__":
    main()
