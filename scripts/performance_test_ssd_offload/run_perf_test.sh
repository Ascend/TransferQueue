#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
PERFTEST_PY="${SCRIPT_DIR}/perftest.py"
CONFIG_YAML="${SCRIPT_DIR}/perftest_config.yaml"

: "${HEAD_NODE_IP:?Set HEAD_NODE_IP to the Ray head node IP}"
: "${WORKER_NODE_IP:?Set WORKER_NODE_IP to the Ray worker node IP}"
: "${SSD_OFFLOAD_PATH:?Set SSD_OFFLOAD_PATH to node-local SSD storage}"

NUM_TEST_ITERATIONS="${NUM_TEST_ITERATIONS:-4}"

mkdir -p "${RESULTS_DIR}"

python "${PERFTEST_PY}" \
    --backend_config="${CONFIG_YAML}" \
    --head_node_ip="${HEAD_NODE_IP}" \
    --worker_node_ip="${WORKER_NODE_IP}" \
    --ssd_path="${SSD_OFFLOAD_PATH}" \
    --output_dir="${RESULTS_DIR}" \
    --num_test_iterations="${NUM_TEST_ITERATIONS}"
