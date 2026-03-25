#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PERFTEST_PY="${SCRIPT_DIR}/perftest.py"
RAY_PERFTEST_PY="${SCRIPT_DIR}/ray_perftest_baseline.py"
CONFIG_YAML="${SCRIPT_DIR}/perftest_config.yaml"

mkdir -p "${RESULTS_DIR}"

# ========== User Configuration ==========
# Modify these based on your environment
HEAD_NODE_IP="${HEAD_NODE_IP:-127.0.0.1}"
WORKER_NODE_IP="${WORKER_NODE_IP:-127.0.0.1}"
DEVICE="${DEVICE:-cpu}"
NUM_TEST_ITERATIONS="${NUM_TEST_ITERATIONS:-4}"
# ========================================

# Backends to test (passed via --backend to perftest.py)
BACKENDS=("SimpleStorage" "Yuanrong" "MooncakeStore")

# Test settings: global_batch_size, field_num, seq_len, name
declare -a SETTINGS=(
    "128,3,1024,Small"
    "1024,9,8192,Medium"
    "4096,21,128000,Large"
)

# ---- TransferQueue perftest ----
for backend in "${BACKENDS[@]}"; do
    echo "=========================================="
    echo "Testing backend: ${backend}"
    echo "=========================================="

    for setting in "${SETTINGS[@]}"; do
        IFS=',' read -r batch_size field_num seq_len name <<< "$setting"
        output_csv="${RESULTS_DIR}/${backend,,}_${name,,}.csv"

        echo "  Setting: ${name} (batch=${batch_size}, fields=${field_num}, seq=${seq_len})"

        python "${PERFTEST_PY}" --backend_config="${CONFIG_YAML}" --backend="${backend}" \
            --device="${DEVICE}" \
            --global_batch_size="${batch_size}" --field_num="${field_num}" --seq_len="${seq_len}" \
            --num_test_iterations="${NUM_TEST_ITERATIONS}" \
            --head_node_ip="${HEAD_NODE_IP}" --worker_node_ip="${WORKER_NODE_IP}" \
            --output_csv="${output_csv}"

        sleep 10
    done
done

# ---- Ray baseline ----
echo "=========================================="
echo "Testing backend: Ray (baseline)"
echo "=========================================="
for setting in "${SETTINGS[@]}"; do
    IFS=',' read -r batch_size field_num seq_len name <<< "$setting"
    output_csv="${RESULTS_DIR}/ray_baseline_${name,,}.csv"

    echo "  Setting: ${name} (batch=${batch_size}, fields=${field_num}, seq=${seq_len})"

    python "${RAY_PERFTEST_PY}" \
        --global_batch_size="${batch_size}" --field_num="${field_num}" --seq_len="${seq_len}" \
        --num_test_iterations="${NUM_TEST_ITERATIONS}" \
        --head_node_ip="${HEAD_NODE_IP}" --worker_node_ip="${WORKER_NODE_IP}" \
        --output_csv="${output_csv}"
done

# ---- Draw figures ----
python "${SCRIPT_DIR}/draw_figure.py"

echo ""
echo "All tests completed!"
