#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PERFTEST_PY="${SCRIPT_DIR}/perftest.py"
CONFIG_YAML="${SCRIPT_DIR}/perftest_config.yaml"

mkdir -p "${RESULTS_DIR}"

# ========== User Configuration ==========
HEAD_NODE_IP="${HEAD_NODE_IP:-127.0.0.1}"
WORKER_NODE_IP="${WORKER_NODE_IP:-127.0.0.1}"
DEVICE="${DEVICE:-cpu}"
NUM_TEST_ITERATIONS="${NUM_TEST_ITERATIONS:-4}"
USE_COMPLEX_CASE="${USE_COMPLEX_CASE:-false}"
SSD_OFFLOAD_PATH="${SSD_OFFLOAD_PATH:-}"
# ========================================

if [[ -z "${SSD_OFFLOAD_PATH}" ]]; then
    echo "SSD_OFFLOAD_PATH must point to an existing local SSD directory." >&2
    exit 1
fi

if [[ ! -d "${SSD_OFFLOAD_PATH}" ]]; then
    echo "SSD_OFFLOAD_PATH does not exist or is not a directory: ${SSD_OFFLOAD_PATH}" >&2
    exit 1
fi

# Extension points for additional SSD-capable backends and workload sizes.
# Currently, SSD offload is a SimpleStorage feature. Workload names describe
# the per-field sample size that controls memory-versus-SSD routing.
BACKENDS=("SimpleStorage")
declare -a SETTINGS=(
    # batch_size, field_num, seq_len, name
    "512,8,262144,Sample1MiB"
    "512,8,524288,Sample2MiB"
    "512,8,1048576,Sample4MiB"
)

COMPLEX_ARGS=()
if [[ "${USE_COMPLEX_CASE}" == "true" ]]; then
    COMPLEX_ARGS=(--use_complex_case)
fi

for backend in "${BACKENDS[@]}"; do
    echo "=========================================="
    echo "Testing SSD offload comparison: ${backend}"
    echo "SSD path: ${SSD_OFFLOAD_PATH}"
    echo "=========================================="

    for setting in "${SETTINGS[@]}"; do
        IFS=',' read -r batch_size field_num seq_len name <<< "${setting}"
        ssd_output_csv="${RESULTS_DIR}/${backend,,}_ssd_${name,,}.csv"
        memory_output_csv="${RESULTS_DIR}/${backend,,}_${name,,}.csv"

        echo "  SSD offload: ${name} (batch=${batch_size}, fields=${field_num}, seq=${seq_len})"
        python "${PERFTEST_PY}" --backend_config="${CONFIG_YAML}" --backend="${backend}" \
            --device="${DEVICE}" \
            --global_batch_size="${batch_size}" --field_num="${field_num}" --seq_len="${seq_len}" \
            --num_test_iterations="${NUM_TEST_ITERATIONS}" \
            --head_node_ip="${HEAD_NODE_IP}" --worker_node_ip="${WORKER_NODE_IP}" \
            --output_csv="${ssd_output_csv}" --ssd_offload --ssd_path="${SSD_OFFLOAD_PATH}" \
            "${COMPLEX_ARGS[@]}"

        sleep 10

        echo "  Host memory: ${name} (batch=${batch_size}, fields=${field_num}, seq=${seq_len})"
        python "${PERFTEST_PY}" --backend_config="${CONFIG_YAML}" --backend="${backend}" \
            --device="${DEVICE}" \
            --global_batch_size="${batch_size}" --field_num="${field_num}" --seq_len="${seq_len}" \
            --num_test_iterations="${NUM_TEST_ITERATIONS}" \
            --head_node_ip="${HEAD_NODE_IP}" --worker_node_ip="${WORKER_NODE_IP}" \
            --output_csv="${memory_output_csv}" \
            "${COMPLEX_ARGS[@]}"

        sleep 10
    done
done

echo ""
echo "All SSD offload comparison tests completed!"
