#!/bin/bash
# A/B breakdown comparison: Baseline (GPU data, CPU memcpy) vs GDR (staging buffer).
#
# Runs breakdown_test.py twice with GPU data:
#   1. Baseline config (use_gdr=false) — measures D2H + host RDMA overhead
#   2. GDR config      (use_gdr=true)  — measures D2D + GDR RDMA
#
# Then generates a side-by-side comparison breakdown chart.
#
# Prerequisites:
#   - 2-node Ray cluster with GPU, RDMA, nvidia_peermem loaded
#   - mooncake-transfer-engine with CUDA support
#
# Usage:
#   ./run_gdr_breakdown_comparison.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
BREAKDOWN_PY="${SCRIPT_DIR}/breakdown_test.py"
BASELINE_CONFIG="${SCRIPT_DIR}/mooncake_config.yaml"
GDR_CONFIG="${SCRIPT_DIR}/mooncake_gdr_config.yaml"
COMPARISON_PY="${RESULTS_DIR}/plot_gdr_breakdown_comparison.py"

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

SCALES="${GDR_BREAKDOWN_SCALES:-small medium}"
NUM_WARMUP="${GDR_BREAKDOWN_WARMUP:-1}"
NUM_ITERS="${GDR_BREAKDOWN_ITERS:-3}"

echo ""
echo "============================================"
echo " Phase 1: Baseline (GPU data, use_gdr=false)"
echo "============================================"
python "${BREAKDOWN_PY}" \
    --backend_config="${BASELINE_CONFIG}" \
    --head_node_ip="${HEAD_IP}" \
    --worker_node_ip="${WORKER_IP}" \
    --device=gpu \
    --num_warmup="${NUM_WARMUP}" \
    --num_iters="${NUM_ITERS}" \
    --scales ${SCALES} \
    --output_dir="${RESULTS_DIR}"

# Rename output files to baseline
mv "${RESULTS_DIR}/breakdown_results.csv"          "${RESULTS_DIR}/breakdown_baseline_gpu.csv"
mv "${RESULTS_DIR}/breakdown_raw_tqtiming.log"     "${RESULTS_DIR}/breakdown_baseline_gpu_raw.log"
mv "${RESULTS_DIR}/breakdown_chart.png"             "${RESULTS_DIR}/breakdown_baseline_gpu_chart.png"

sleep 10

echo ""
echo "============================================"
echo " Phase 2: GDR (GPU data, use_gdr=true)"
echo "============================================"
python "${BREAKDOWN_PY}" \
    --backend_config="${GDR_CONFIG}" \
    --head_node_ip="${HEAD_IP}" \
    --worker_node_ip="${WORKER_IP}" \
    --device=gpu \
    --num_warmup="${NUM_WARMUP}" \
    --num_iters="${NUM_ITERS}" \
    --scales ${SCALES} \
    --output_dir="${RESULTS_DIR}"

# Rename output files to GDR
mv "${RESULTS_DIR}/breakdown_results.csv"          "${RESULTS_DIR}/breakdown_gdr_gpu.csv"
mv "${RESULTS_DIR}/breakdown_raw_tqtiming.log"     "${RESULTS_DIR}/breakdown_gdr_gpu_raw.log"
mv "${RESULTS_DIR}/breakdown_chart.png"             "${RESULTS_DIR}/breakdown_gdr_gpu_chart.png"

echo ""
echo "============================================"
echo " Phase 3: Comparison Chart"
echo "============================================"
python "${COMPARISON_PY}" \
    --baseline="${RESULTS_DIR}/breakdown_baseline_gpu.csv" \
    --gdr="${RESULTS_DIR}/breakdown_gdr_gpu.csv" \
    --scales ${SCALES} \
    --output="${RESULTS_DIR}/gdr_breakdown_comparison.png"

echo ""
echo "============================================"
echo " Results in: ${RESULTS_DIR}/"
echo "   breakdown_baseline_gpu.csv        (baseline breakdown data)"
echo "   breakdown_gdr_gpu.csv             (GDR breakdown data)"
echo "   breakdown_baseline_gpu_chart.png  (baseline breakdown chart)"
echo "   breakdown_gdr_gpu_chart.png       (GDR breakdown chart)"
echo "   gdr_breakdown_comparison.png      (side-by-side comparison)"
echo "============================================"
