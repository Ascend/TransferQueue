#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python3}"
"$python_bin" -m compileall -q transfer_queue tutorial tests
"$python_bin" -m pytest -q tests/test_controller_fetch_progress.py
