#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
    smoke|pilot) ;;
    *) echo "Usage: bash meanflow_mp1_ready/run.sh [smoke|pilot]" >&2; exit 2 ;;
esac
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$BUNDLE_DIR/.." && pwd)"
cd -- "$PROJECT"

python "$BUNDLE_DIR/install.py" --project "$PROJECT"
DATASET="${MF_DATASET:-$PROJECT/data/robomimic/datasets/kitchen_d1/kitchen_d1_voxel_abs.hdf5}"
if [[ ! -f "$DATASET" ]]; then
    echo "Dataset missing: $DATASET. Set MF_DATASET to the training HDF5 path." >&2
    exit 2
fi
if [[ -n "${MF_CHECKPOINT:-}" ]]; then
    CKPT="$MF_CHECKPOINT"
else
    CKPT="$(python scripts/meanflow_experiment.py find-checkpoint \
        --root data/outputs --task kitchen_d1 --training-seed 20 --n-demo 200)"
fi
if [[ ! -f "$CKPT" ]]; then
    echo "Checkpoint missing: $CKPT. Set MF_CHECKPOINT to an existing latest.ckpt." >&2
    exit 2
fi
DEVICE="${MF_DEVICE:-cuda:0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
OUTPUT="${MF_OUTPUT:-$PROJECT/data/mf_kitchen_${MODE}_$(date +%Y%m%d_%H%M%S)}"

echo "MP1 policy: $PROJECT/policy/mp1.py"
echo "Checkpoint: $CKPT"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
python scripts/meanflow_experiment.py doctor \
    --checkpoint "$CKPT" --dataset-path "$DATASET" --device "$DEVICE"

COMMON=(--backend robomimic --checkpoint "$CKPT" --dataset-path "$DATASET"
        --output "$OUTPUT" --device "$DEVICE" --phase all --steps 1 2 4)
if [[ "$MODE" == smoke ]]; then
    python -u scripts/meanflow_experiment.py run "${COMMON[@]}" \
        --episodes 2 --trials 1 --schedules uniform --anchor-every 1 --max-anchors 2
else
    python -u scripts/meanflow_experiment.py run "${COMMON[@]}" \
        --episodes 20 --trials 3 --schedules uniform noise_first data_first \
        --anchor-every 5 --max-anchors 3
fi
echo "Completed. Read: $OUTPUT/report.md"
