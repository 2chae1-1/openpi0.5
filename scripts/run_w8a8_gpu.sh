#!/usr/bin/env bash
# Run from the repository root: bash scripts/run_w8a8_gpu.sh
# Set these environment variables or edit the defaults below. See GPU_HANDOFF.md.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_ROOT"

# Relative paths are resolved from the repository root. PYTHON_BIN is one executable.
W8A8_CHECKPOINT="${W8A8_CHECKPOINT:-checkpoints/pi05-libero-w8a8-public}"
FP_CHECKPOINT="${FP_CHECKPOINT:-}" # Empty means W8A8 only; no FP worker is launched.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/w8a8-gpu}"
RUNS="${RUNS:-3}"
DEVICE="${DEVICE:-cuda:0}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-50}"
NUM_STEPS="${NUM_STEPS:-10}"
SEED="${SEED:-42}" # Keep the same input/noise across independent process runs.
PROMPT="${PROMPT:-pick up the book}"
OBSERVATION="${OBSERVATION:-}" # Optional recorded LIBERO NPZ; otherwise synthetic input.
W8A8_REVISION="${W8A8_REVISION:-}" # Declare only a revision you actually verified.
FP_REVISION="${FP_REVISION:-}"
ENABLE_TUNING="${ENABLE_TUNING:-0}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

if [[ $# -ne 0 ]]; then
    if [[ $# -eq 1 && "$1" == "--help" ]]; then
        cat <<'HELP'
Usage: bash scripts/run_w8a8_gpu.sh
Configure with environment variables (paths may contain spaces):
  W8A8_CHECKPOINT  default: checkpoints/pi05-libero-w8a8-public
  PYTHON_BIN      default: .venv/bin/python (one executable, without arguments)
  FP_CHECKPOINT   default: empty (W8A8 only); set to enable optional FP comparison
  RUNS            default: 3 independent experiments
  OUTPUT_ROOT     default: data/w8a8-gpu (a unique batch directory is created)
  DEVICE          default: cuda:0
  WARMUP=10 ITERATIONS=50 NUM_STEPS=10 SEED=42
  PROMPT='pick up the book' OBSERVATION='' ENABLE_TUNING=0
  W8A8_REVISION='' FP_REVISION='' (optional operator-declared source revisions)
All relative paths use the repository root. See GPU_HANDOFF.md for preparation.
HELP
        exit 0
    fi
    die "Use environment variables for settings; only --help is accepted."
fi

[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer without leading zeros."
[[ "$ENABLE_TUNING" == 0 || "$ENABLE_TUNING" == 1 ]] || die "ENABLE_TUNING must be 0 or 1."
[[ "$DEVICE" =~ ^cuda(:[0-9]+)?$ ]] || die "DEVICE must be cuda or cuda:N; CPU fallback is disabled."
command -v -- "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
command -v tee >/dev/null 2>&1 || die "tee is required to preserve console logs."
# The Python CLI silently skips missing FP weights. Make an explicit comparison request fail clearly.
if [[ -n "$FP_CHECKPOINT" && ( ! -f "$FP_CHECKPOINT/model.safetensors" || ! -s "$FP_CHECKPOINT/model.safetensors" ) ]]; then
    die "FP_CHECKPOINT requires a nonempty local model.safetensors: $FP_CHECKPOINT"
fi

mkdir -p -- "$OUTPUT_ROOT"
# mktemp prevents collisions even for concurrent launches within the same second.
BATCH_DIR="$(mktemp -d "$OUTPUT_ROOT/batch-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
BATCH_DIR="$(cd -- "$BATCH_DIR" && pwd)"
printf 'Repository: %s\nResults: %s\nExperiments: %s\n' "$REPO_ROOT" "$BATCH_DIR" "$RUNS"

for ((run = 1; run <= RUNS; run++)); do
    printf -v run_name 'run_%02d' "$run"
    run_dir="$BATCH_DIR/$run_name"
    mkdir -- "$run_dir"
    # Keep launcher files outside reports/: the Python CLI requires a new/empty output directory.
    cmd=(
        "$PYTHON_BIN" -B -u "$SCRIPT_DIR/benchmark_pi05_fp_w8a8.py"
        --w8a8-checkpoint "$W8A8_CHECKPOINT"
        --output-dir "$run_dir/reports"
        --device "$DEVICE"
        --warmup "$WARMUP"
        --iterations "$ITERATIONS"
        --num-steps "$NUM_STEPS"
        --seed "$SEED"
        --prompt "$PROMPT"
    )
    [[ -z "$FP_CHECKPOINT" ]] || cmd+=(--fp-checkpoint "$FP_CHECKPOINT")
    [[ -z "$OBSERVATION" ]] || cmd+=(--observation "$OBSERVATION")
    [[ -z "$W8A8_REVISION" ]] || cmd+=(--w8a8-revision "$W8A8_REVISION")
    [[ -z "$FP_REVISION" ]] || cmd+=(--fp-revision "$FP_REVISION")
    [[ "$ENABLE_TUNING" == 0 ]] || cmd+=(--enable-tuning)
    {
        printf '# Run from repository: %s\n' "$REPO_ROOT"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$run_dir/command.txt"
    printf '\n[%s/%s] %s\n' "$run" "$RUNS" "$run_dir" | tee "$run_dir/console.log"

    # Capture both statuses immediately; tee must never hide a failed Python process.
    set +e
    "${cmd[@]}" 2>&1 | tee -a "$run_dir/console.log"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    python_status="${pipeline_status[0]}"
    tee_status="${pipeline_status[1]}"
    printf '%s\n' "$python_status" > "$run_dir/exit_code.txt"

    if ((python_status != 0)); then
        printf 'ERROR: experiment %s failed (Python exit %s). Stopping; no FP fallback.\nInspect: %s\n' \
            "$run_name" "$python_status" "$run_dir" | tee -a "$run_dir/console.log" >&2 || true
        exit "$python_status"
    fi
    if ((tee_status != 0)); then
        printf 'ERROR: console log write failed for %s (tee exit %s).\n' "$run_dir" "$tee_status" >&2
        exit "$tee_status"
    fi
done

printf '\nAll %s benchmark processes exited successfully. Results: %s\n' "$RUNS" "$BATCH_DIR"
