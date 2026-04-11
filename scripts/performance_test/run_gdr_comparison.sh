#!/bin/bash
# A/B comparison: MooncakeStore with vs without GDR.
#
# Usage:
#   ./run_gdr_comparison.sh [extra args for perftest.py]
#
# Prerequisites:
#   - 2-node Ray cluster with GPU, RDMA, nvidia_peermem loaded
#   - mooncake-transfer-engine built with USE_CUDA=ON
#
# Output: two CSV files in results/ — one for baseline, one for GDR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PERFTEST_PY="${SCRIPT_DIR}/perftest.py"
BASELINE_CONFIG="${SCRIPT_DIR}/mooncake_config.yaml"
GDR_CONFIG="${SCRIPT_DIR}/mooncake_gdr_config.yaml"

mkdir -p "${RESULTS_DIR}"

# Auto-detect head and worker IPs from Ray cluster
read -r HEAD_IP WORKER_IP <<< $(python3 -c "
import ray
ray.init(address='auto', logging_level='WARNING')
nodes = ray.nodes()
head = worker = None
for n in nodes:
    addr = n['NodeManagerAddress']
    res = n['Resources']
    if 'node:__internal_head__' in res:
        for k in res:
            if k.startswith('node:') and k != 'node:__internal_head__':
                head = k.split('node:', 1)[1]
                break
    else:
        worker = addr
if head is None:
    print('ERROR: head node not found', flush=True)
    exit(1)
if worker is None:
    worker = head
print(head, worker)
")

echo "Head node:   $HEAD_IP"
echo "Worker node: $WORKER_IP"

NUM_ITERS="${NUM_TEST_ITERATIONS:-4}"

# Test matrix: batch_size, field_num, seq_len, label
SETTINGS=(
    "1024,9,8192,Small"
    "4096,15,32768,Medium"
)

for setting in "${SETTINGS[@]}"; do
    IFS=',' read -r batch_size field_num seq_len label <<< "$setting"
    data_desc="batch=${batch_size},fields=${field_num},seq=${seq_len}"

    echo ""
    echo "============================================"
    echo " ${label}: ${data_desc}"
    echo "============================================"

    # --- Baseline (no GDR) ---
    echo ">>> Baseline (use_gdr=false)"
    python "${PERFTEST_PY}" \
        --backend_config="${BASELINE_CONFIG}" \
        --backend=MooncakeStore \
        --device=gpu \
        --global_batch_size="${batch_size}" \
        --field_num="${field_num}" \
        --seq_len="${seq_len}" \
        --num_test_iterations="${NUM_ITERS}" \
        --head_node_ip="${HEAD_IP}" \
        --worker_node_ip="${WORKER_IP}" \
        --output_csv="${RESULTS_DIR}/mooncake_baseline_${label,,}.csv" \
        "$@"

    sleep 5

    # --- GDR ---
    echo ">>> GDR (use_gdr=true)"
    python "${PERFTEST_PY}" \
        --backend_config="${GDR_CONFIG}" \
        --backend=MooncakeStore \
        --device=gpu \
        --global_batch_size="${batch_size}" \
        --field_num="${field_num}" \
        --seq_len="${seq_len}" \
        --num_test_iterations="${NUM_ITERS}" \
        --head_node_ip="${HEAD_IP}" \
        --worker_node_ip="${WORKER_IP}" \
        --output_csv="${RESULTS_DIR}/mooncake_gdr_${label,,}.csv" \
        "$@"

    sleep 5
done

echo ""
echo "============================================"
echo " Results in: ${RESULTS_DIR}/"
echo "   mooncake_baseline_*.csv  (no GDR)"
echo "   mooncake_gdr_*.csv       (with GDR)"
echo "============================================"
echo ""
echo "Compare [TQ-TIMING] lines in stdout:"
echo "  Baseline: mooncake::batch_put_tensor  → bw=...Gb/s"
echo "  GDR:      mooncake::gdr_batch_put     → bw=...Gb/s"
