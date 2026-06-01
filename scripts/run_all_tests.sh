#!/usr/bin/env bash
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

# run_all_tests.sh: full local mirror of TransferQueue CI.
#
# Mirrors the four CI workflows in .github/workflows/:
#   1. sanity        (license headers + docstrings)
#   2. unit          (pytest tests/  — default backend = SimpleStorage)
#   3. e2e-backends  (Mooncake / Yuanrong e2e suites — optional)
#   4. tutorials     (tutorial/*.py + tutorial/basic.ipynb)
#   5. recipes       (recipe/simple_use_case/*.py)
#
# Each phase logs to _test_logs/<phase>.log and contributes a row to a
# pass/fail summary at the end. A phase failure does not abort the run —
# every phase is attempted so you see the full surface in one go.
#
# Usage:
#   scripts/run_all_tests.sh [options]
#
# Options:
#   --python <bin>          Python interpreter to seed the venv (default: python3.11
#                           if present, else python3).
#   --venv <path>           Venv location (default: ./.venv-tq-tests).
#   --no-install            Skip venv creation / dep install. Use the current
#                           interpreter. Useful for fast re-runs.
#   --only <list>           Comma-separated phases to run. Choices:
#                           sanity, unit, mooncake, yuanrong, tutorials, recipes.
#                           Default: sanity,unit,tutorials,recipes
#                           (mooncake and yuanrong require extra binaries —
#                            opt in explicitly).
#   --skip <list>           Comma-separated phases to skip.
#   --keep-venv             Keep the venv on exit (default: keep on success,
#                           keep on failure too — we never auto-delete).
#   --recreate-venv         Force-delete an existing venv before install.
#   --no-color              Disable ANSI color output.
#   -h, --help              Show this help.
#
# Environment variables honored:
#   TQ_TESTS_PYTHON         Same as --python.
#   TQ_TESTS_VENV           Same as --venv.
#   TQ_NUM_THREADS          Forwarded to tutorials/recipes (default: 2).
#   RAY_DEDUP_LOGS          Forwarded (default: 0).
#   PYTEST_ADDOPTS          Extra pytest args (e.g. "-x -k metadata").
#
# Ray daemons leak across processes (raylet, gcs_server, plasma_store, dashboard,
# worker pools). To avoid "repeat init" / port-6379-conflict errors between
# tutorials, recipes, and pytest runs, the script:
#   - runs `ray stop --force` + pkill before each Ray-using invocation
#   - runs the same cleanup after each invocation
#   - registers an EXIT/INT/TERM trap so Ctrl-C never leaves orphans behind
#
# Exit code: 0 iff every selected phase passed.

set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/_test_logs"

DEFAULT_PHASES="sanity,unit,tutorials,recipes"
OPTIONAL_PHASES="mooncake,yuanrong"

PYTHON_BIN="${TQ_TESTS_PYTHON:-}"
VENV_PATH="${TQ_TESTS_VENV:-${REPO_ROOT}/.venv-tq-tests}"
ONLY=""
SKIP=""
NO_INSTALL=0
RECREATE_VENV=0
NO_COLOR=0

export TQ_NUM_THREADS="${TQ_NUM_THREADS:-2}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

usage() {
    sed -n '17,64p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)        PYTHON_BIN="$2"; shift 2 ;;
        --venv)          VENV_PATH="$2";  shift 2 ;;
        --only)          ONLY="$2";       shift 2 ;;
        --skip)          SKIP="$2";       shift 2 ;;
        --no-install)    NO_INSTALL=1;    shift ;;
        --recreate-venv) RECREATE_VENV=1; shift ;;
        --keep-venv)     shift ;; # accepted for symmetry, no-op (we keep by default)
        --no-color)      NO_COLOR=1;      shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
if [[ $NO_COLOR -eq 0 && -t 1 ]]; then
    C_RED=$'\033[0;31m'
    C_GREEN=$'\033[0;32m'
    C_YELLOW=$'\033[0;33m'
    C_BLUE=$'\033[0;34m'
    C_BOLD=$'\033[1m'
    C_DIM=$'\033[2m'
    C_RESET=$'\033[0m'
else
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""; C_DIM=""; C_RESET=""
fi

banner() {
    echo
    echo "${C_BOLD}${C_BLUE}===============================================================${C_RESET}"
    echo "${C_BOLD}${C_BLUE}  $1${C_RESET}"
    echo "${C_BOLD}${C_BLUE}===============================================================${C_RESET}"
}

info()    { echo "${C_DIM}[info]${C_RESET}    $*"; }
warn()    { echo "${C_YELLOW}[warn]${C_RESET}    $*"; }
fail()    { echo "${C_RED}[fail]${C_RESET}    $*"; }
success() { echo "${C_GREEN}[ ok ]${C_RESET}    $*"; }

# ---------------------------------------------------------------------------
# Ray cleanup
# ---------------------------------------------------------------------------
# Ray spawns persistent daemons (raylet, gcs_server, plasma_store, ray::IDLE
# workers, dashboard, …) that survive the parent Python process. If we don't
# nuke them between subprocess invocations, the next tutorial/recipe/pytest
# either reconnects to the stale cluster, hits port conflicts on 6379/8265,
# or raises "ray has already been initialized" / "Maybe you called ray.init
# twice by accident". This helper is best-effort and idempotent.
#
# We always:
#   1. Try `ray stop --force` if the CLI is on PATH (clean shutdown via Ray).
#   2. Fall back to pkill for any leftover daemon processes (works even if
#      the ray CLI is missing or hung).
#   3. Clear inherited RAY_ADDRESS so a child doesn't try to attach to a
#      non-existent cluster.
#   4. Brief sleep so OS releases ports / sockets before the next invocation.
cleanup_ray() {
    if command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
    fi
    # Catch every Ray-spawned daemon we've seen leak in practice.
    pkill -9 -f '[r]aylet'        2>/dev/null || true
    pkill -9 -f '[g]cs_server'    2>/dev/null || true
    pkill -9 -f '[p]lasma_store'  2>/dev/null || true
    pkill -9 -f '[r]ay::'         2>/dev/null || true
    pkill -9 -f '[d]ashboard.*ray' 2>/dev/null || true
    pkill -9 -f '[r]ay.*default_worker' 2>/dev/null || true
    unset RAY_ADDRESS
    sleep 1
}

# Ensure we never leave Ray daemons running when the script exits, even on
# Ctrl-C or an unexpected error. The trap is set after the helper is defined.
on_exit() {
    local rc=$?
    cleanup_ray
    exit $rc
}
trap on_exit EXIT
trap 'echo; warn "interrupted, cleaning up Ray daemons"; exit 130' INT TERM

# ---------------------------------------------------------------------------
# Phase selection (bash 3.2 compatible — macOS /bin/bash has no `declare -A`)
# ---------------------------------------------------------------------------
all_known="sanity unit mooncake yuanrong tutorials recipes"

if [[ -n "$ONLY" ]]; then
    selected="$ONLY"
else
    selected="$DEFAULT_PHASES"
fi

# Track enabled phases as a space-padded string for membership tests.
ENABLED=" "
for p in $(echo "$selected" | tr ',' ' '); do
    [[ -z "$p" ]] && continue
    case " $all_known " in
        *" $p "*) ;;
        *) echo "Unknown phase '$p'. Known: $all_known" >&2; exit 2 ;;
    esac
    case "$ENABLED" in *" $p "*) ;; *) ENABLED+="$p " ;; esac
done

if [[ -n "$SKIP" ]]; then
    for p in $(echo "$SKIP" | tr ',' ' '); do
        [[ -z "$p" ]] && continue
        ENABLED="${ENABLED// $p / }"
    done
fi

is_enabled() {
    case "$ENABLED" in *" $1 "*) return 0 ;; esac
    return 1
}

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

if [[ $NO_INSTALL -eq 1 ]]; then
    if [[ -z "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3 || true)"
    fi
    if [[ -z "$PYTHON_BIN" ]]; then
        echo "No python3 found on PATH. Pass --python." >&2
        exit 2
    fi
    PY="$PYTHON_BIN"
    PIP="$PYTHON_BIN -m pip"
    info "Using existing interpreter: $PY (skipping install)"
else
    if [[ -z "$PYTHON_BIN" ]]; then
        PYTHON_BIN="$(command -v python3.11 || command -v python3 || true)"
    fi
    if [[ -z "$PYTHON_BIN" ]]; then
        echo "No python3.11 / python3 found. Pass --python." >&2
        exit 2
    fi

    if [[ $RECREATE_VENV -eq 1 && -d "$VENV_PATH" ]]; then
        info "Removing existing venv at $VENV_PATH"
        rm -rf "$VENV_PATH"
    fi

    if [[ ! -d "$VENV_PATH" ]]; then
        banner "Creating venv at $VENV_PATH using $PYTHON_BIN"
        if command -v uv >/dev/null 2>&1; then
            uv venv --python "$PYTHON_BIN" "$VENV_PATH"
        else
            "$PYTHON_BIN" -m venv "$VENV_PATH"
        fi
    else
        info "Reusing existing venv at $VENV_PATH"
    fi

    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
    PY="$(command -v python)"
    PIP="$PY -m pip"

    banner "Installing dependencies"
    $PIP install --upgrade pip
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    $PIP install -e "$REPO_ROOT[test,build,yuanrong]"

    if is_enabled mooncake; then
        info "Installing mooncake-transfer-engine-non-cuda for Mooncake backend"
        $PIP install mooncake-transfer-engine-non-cuda || \
            warn "mooncake-transfer-engine-non-cuda install failed; Mooncake e2e will be skipped"
    fi

    if is_enabled tutorials; then
        $PIP install nbconvert ipykernel
    fi
fi

# Nuke any leftover Ray daemons from a previously-crashed run before we
# print versions or start any phase. Otherwise the first Ray-using phase
# silently inherits a stale cluster on 6379/8265.
banner "Pre-flight Ray cleanup"
cleanup_ray
info "Killed any stale Ray daemons from previous runs"

# Numpy + TransferQueue sanity check so we know exactly what we're testing.
banner "Environment versions"
$PY - <<'PY'
import importlib
import sys

print(f"python:         {sys.version.split()[0]}")
for mod in ("numpy", "torch", "ray", "tensordict", "transfer_queue", "msgspec", "hydra", "omegaconf"):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
    except Exception as e:
        ver = f"<not installed: {e.__class__.__name__}>"
    print(f"{mod:15s} {ver}")
PY

# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------
# results: phase|status|elapsed-seconds|log-path
declare -a RESULTS=()
OVERALL_RC=0

run_phase() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    banner "PHASE: $name"
    info "Logging to $logfile"
    local start end elapsed rc
    start=$(date +%s)

    # The first arg is the command — we tee to log so user sees live output.
    (
        set -o pipefail
        "$@" 2>&1
    ) | tee "$logfile"
    rc=${PIPESTATUS[0]}

    end=$(date +%s)
    elapsed=$((end - start))

    if [[ $rc -eq 0 ]]; then
        success "$name passed in ${elapsed}s"
        RESULTS+=("$name|PASS|${elapsed}|$logfile")
    else
        fail "$name FAILED (rc=$rc) after ${elapsed}s — see $logfile"
        RESULTS+=("$name|FAIL($rc)|${elapsed}|$logfile")
        OVERALL_RC=1
    fi
}

# Some phases want pkill cleanup between sub-runs (mooncake_master daemons).
cleanup_mooncake_master() {
    pkill -f '[m]ooncake_master' 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------
phase_sanity() {
    cd "$REPO_ROOT"
    $PY tests/sanity/check_license.py --directories . && \
        $PY tests/sanity/check_docstrings.py
}

phase_unit() {
    cd "$REPO_ROOT"
    # Excludes the e2e/ dir from the default run — those have their own phase
    # variants for the Mooncake / Yuanrong backends. Run the SimpleStorage-default
    # e2e here so the default backend is still exercised.
    cleanup_ray
    $PY -m pytest tests \
        --ignore=tests/sanity \
        ${PYTEST_ADDOPTS:-}
    local rc=$?
    cleanup_ray
    return $rc
}

phase_mooncake() {
    cd "$REPO_ROOT"
    if ! $PY -c "import mooncake" 2>/dev/null && \
       ! $PY -c "import mooncake_master" 2>/dev/null; then
        # The package name varies — try a friendlier check.
        if ! $PIP show mooncake-transfer-engine-non-cuda >/dev/null 2>&1 && \
           ! $PIP show mooncake-transfer-engine >/dev/null 2>&1; then
            warn "mooncake-transfer-engine is not installed; skipping."
            return 0
        fi
    fi
    local rc1 rc2
    cleanup_ray
    cleanup_mooncake_master
    TQ_TEST_BACKEND=MooncakeStore $PY -m pytest tests/e2e/test_e2e_lifecycle_consistency.py \
        ${PYTEST_ADDOPTS:-}
    rc1=$?
    cleanup_ray
    cleanup_mooncake_master
    TQ_TEST_BACKEND=MooncakeStore $PY -m pytest tests/e2e/test_kv_interface_e2e.py \
        ${PYTEST_ADDOPTS:-}
    rc2=$?
    cleanup_ray
    cleanup_mooncake_master
    return $(( rc1 != 0 ? rc1 : rc2 ))
}

phase_yuanrong() {
    cd "$REPO_ROOT"
    if ! $PIP show openyuanrong-datasystem >/dev/null 2>&1; then
        warn "openyuanrong-datasystem is not installed; skipping."
        return 0
    fi
    local rc1 rc2
    cleanup_ray
    TQ_TEST_BACKEND=Yuanrong $PY -m pytest tests/e2e/test_e2e_lifecycle_consistency.py \
        ${PYTEST_ADDOPTS:-}
    rc1=$?
    cleanup_ray
    TQ_TEST_BACKEND=Yuanrong $PY -m pytest tests/e2e/test_kv_interface_e2e.py \
        ${PYTEST_ADDOPTS:-}
    rc2=$?
    cleanup_ray
    return $(( rc1 != 0 ? rc1 : rc2 ))
}

phase_tutorials() {
    cd "$REPO_ROOT"
    local failed=0
    cleanup_ray
    for f in tutorial/*.py; do
        info "Running tutorial: $f"
        if ! $PY "$f"; then
            fail "tutorial failed: $f"
            failed=$((failed + 1))
        fi
        # Every tutorial spawns its own Ray cluster — tear it down before
        # starting the next so we don't see port-conflict / repeat-init errors.
        cleanup_ray
    done
    if command -v jupyter >/dev/null 2>&1; then
        info "Running notebook tutorial: tutorial/basic.ipynb"
        if ! jupyter nbconvert --to notebook --execute \
            --ExecutePreprocessor.timeout=120 \
            --output /tmp/tq_basic_executed.ipynb \
            tutorial/basic.ipynb; then
            fail "notebook tutorial failed: tutorial/basic.ipynb"
            failed=$((failed + 1))
        fi
        cleanup_ray
    else
        warn "jupyter not on PATH; skipping notebook tutorial."
    fi
    return $failed
}

phase_recipes() {
    cd "$REPO_ROOT"
    local rc1 rc2
    cleanup_ray
    $PY recipe/simple_use_case/single_controller_demo.py \
        --num-samples 8 --global-batch-size 4 --rollout-agent-num-workers 1
    rc1=$?
    cleanup_ray
    $PY recipe/simple_use_case/relax_demo.py \
        --num-steps 1 --global-batch-size 1 --micro-batch-size 1 \
        --num-rollout-workers 1 --num-ref-workers 1 \
        --num-actor-workers 1 --num-reward-workers 1 \
        --rollout-sleep-seconds 0.01 --stage-sleep-seconds 0.01 \
        --weight-sync-seconds 0.01
    rc2=$?
    cleanup_ray
    return $(( rc1 != 0 ? rc1 : rc2 ))
}

# ---------------------------------------------------------------------------
# Run!
# ---------------------------------------------------------------------------
is_enabled sanity    && run_phase sanity    phase_sanity
is_enabled unit      && run_phase unit      phase_unit
is_enabled mooncake  && run_phase mooncake  phase_mooncake
is_enabled yuanrong  && run_phase yuanrong  phase_yuanrong
is_enabled tutorials && run_phase tutorials phase_tutorials
is_enabled recipes   && run_phase recipes   phase_recipes

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner "Summary"
printf '%-12s  %-10s  %8s  %s\n' "PHASE" "STATUS" "ELAPSED" "LOG"
printf '%-12s  %-10s  %8s  %s\n' "-----" "------" "-------" "---"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r name status elapsed logfile <<< "$row"
    if [[ "$status" == "PASS" ]]; then
        color="$C_GREEN"
    else
        color="$C_RED"
    fi
    printf "%-12s  ${color}%-10s${C_RESET}  %7ss  %s\n" \
        "$name" "$status" "$elapsed" "$logfile"
done

if [[ $OVERALL_RC -eq 0 ]]; then
    echo
    echo "${C_GREEN}${C_BOLD}All selected phases passed.${C_RESET}"
else
    echo
    echo "${C_RED}${C_BOLD}One or more phases failed. See per-phase logs above.${C_RESET}"
fi

exit $OVERALL_RC
