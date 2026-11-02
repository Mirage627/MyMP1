#!/usr/bin/env bash
# Run from the project holding data/, with this source package in EfficientFlow/.
set -euo pipefail

mp1_usage() {
  cat <<'EOF'
Usage: bash EfficientFlow/scripts/run_mp1_ablation_server.sh [all|CONFIG] [--dry-run]
CONFIG: MP1_baseline MP1_mgad MP1_diag MP1_full MP1_mgad_diag MP1_mgad_full

Environment defaults:
  MP1_PROJECT_ROOT=<parent of source package> MP1_PROFILE=mp1
  MP1_TASK=kitchen_d1 MP1_N_DEMO=200 MP1_SEED=20 MP1_GPU=0
  MP1_DATASET=<project>/data/robomimic/datasets/<task>/<task>_voxel_abs.hdf5
  MP1_RUN_ROOT=<project>/runs MP1_RESUME=false MP1_PYTHON=python
  MP1_BATCH_SIZE=128 MP1_WORKERS=8 MP1_N_ENVS=28 MP1_PROJECT=MP1_ablation
  MP1_MAX_TRAIN_STEPS=null MP1_MAX_VAL_STEPS=null
  MP1_N_TRAIN=6 MP1_N_TEST=50 MP1_N_TRAIN_VIS=2 MP1_N_TEST_VIS=4
Optional budget overrides: MP1_EPOCHS, MP1_ROLLOUT_EVERY, MP1_CHECKPOINT_EVERY.
Unset budget values use the selected YAML (200 demos: 250 epochs, rollout/checkpoint every 5).
ROLLOUT_EVERY, CHECKPOINT_EVERY and N_ENVS are accepted as fallback aliases.
All six groups run serially with the same profile, data, seed and budgets.
--dry-run checks paths and prints commands; it does not train or create outputs.
EOF
}

mp1_fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
mp1_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || mp1_fail "$1 must be a positive integer"
}
mp1_nonnegative_integer() {
  [[ "$2" =~ ^(0|[1-9][0-9]*)$ ]] || mp1_fail "$1 must be a nonnegative integer"
}

mp1_requested_group=all
mp1_group_supplied=false
mp1_dry_run=false
for mp1_argument in "$@"; do
  case "$mp1_argument" in
    --help|-h) mp1_usage; exit 0 ;;
    --dry-run) mp1_dry_run=true ;;
    --*) mp1_fail "Unknown option: $mp1_argument" ;;
    *)
      [[ "$mp1_group_supplied" == false ]] || mp1_fail "Specify only one group or all"
      mp1_requested_group="$mp1_argument"
      mp1_group_supplied=true
      ;;
  esac
done

mp1_source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mp1_project_root="${MP1_PROJECT_ROOT:-$(dirname -- "$mp1_source_root")}"
[[ -d "$mp1_project_root" ]] || mp1_fail "Project directory is missing: $mp1_project_root"
mp1_project_root="$(cd -- "$mp1_project_root" && pwd -P)"
mp1_entry="$mp1_source_root/train_mp1_ablation.py"
[[ -f "$mp1_entry" ]] || mp1_fail "Source entry is missing: $mp1_entry"

mp1_profile="${MP1_PROFILE:-mp1}"
mp1_task="${MP1_TASK:-kitchen_d1}"
mp1_n_demo="${MP1_N_DEMO:-200}"
mp1_seed="${MP1_SEED:-20}"
mp1_epochs="${MP1_EPOCHS:-}"
mp1_gpu="${MP1_GPU:-0}"
mp1_batch_size="${MP1_BATCH_SIZE:-128}"
mp1_workers="${MP1_WORKERS:-8}"
mp1_rollout_every="${MP1_ROLLOUT_EVERY:-${ROLLOUT_EVERY:-}}"
mp1_n_envs="${MP1_N_ENVS:-${N_ENVS:-28}}"
mp1_checkpoint_every="${MP1_CHECKPOINT_EVERY:-${CHECKPOINT_EVERY:-}}"
mp1_max_train_steps="${MP1_MAX_TRAIN_STEPS:-null}"
mp1_max_val_steps="${MP1_MAX_VAL_STEPS:-null}"
mp1_n_train="${MP1_N_TRAIN:-6}"
mp1_n_test="${MP1_N_TEST:-50}"
mp1_n_train_vis="${MP1_N_TRAIN_VIS:-2}"
mp1_n_test_vis="${MP1_N_TEST_VIS:-4}"
mp1_resume="${MP1_RESUME:-false}"
mp1_python="${MP1_PYTHON:-python}"
mp1_logging_project="${MP1_PROJECT:-MP1_ablation}"

[[ "$mp1_profile" == mp1 || "$mp1_profile" == theory ]] || mp1_fail "MP1_PROFILE must be mp1 or theory"
[[ "$mp1_resume" == true || "$mp1_resume" == false ]] || mp1_fail "MP1_RESUME must be true or false"
[[ "$mp1_task" =~ ^[A-Za-z0-9_]+$ ]] || mp1_fail "MP1_TASK must be a task identifier"
for mp1_field in n_demo batch_size n_envs n_train n_test; do
  mp1_variable="mp1_${mp1_field}"
  mp1_positive_integer "$mp1_variable" "${!mp1_variable}"
done
for mp1_field in epochs rollout_every checkpoint_every; do
  mp1_variable="mp1_${mp1_field}"
  [[ -z "${!mp1_variable}" ]] || mp1_positive_integer "$mp1_variable" "${!mp1_variable}"
done
for mp1_field in seed gpu workers n_train_vis n_test_vis; do
  mp1_variable="mp1_${mp1_field}"
  mp1_nonnegative_integer "$mp1_variable" "${!mp1_variable}"
done
for mp1_field in max_train_steps max_val_steps; do
  mp1_variable="mp1_${mp1_field}"
  [[ "${!mp1_variable}" == null ]] || mp1_positive_integer "$mp1_variable" "${!mp1_variable}"
done
(( mp1_n_train_vis <= mp1_n_train && mp1_n_test_vis <= mp1_n_test )) || mp1_fail "Video counts cannot exceed rollout counts"
(( mp1_n_train_vis + mp1_n_test_vis > 0 )) || mp1_fail "At least one rollout video is required for this launcher"
command -v "$mp1_python" >/dev/null 2>&1 || mp1_fail "Python executable is unavailable: $mp1_python"

mp1_dataset="${MP1_DATASET:-$mp1_project_root/data/robomimic/datasets/$mp1_task/${mp1_task}_voxel_abs.hdf5}"
[[ "$mp1_dataset" == /* ]] || mp1_dataset="$mp1_project_root/$mp1_dataset"
[[ -d "$(dirname -- "$mp1_dataset")" ]] || mp1_fail "Dataset directory is missing: $(dirname -- "$mp1_dataset")"
[[ -f "$mp1_dataset" ]] || mp1_fail "Dataset file is missing: $mp1_dataset"
mp1_dataset="$(readlink -f -- "$mp1_dataset")"
[[ -r "$mp1_dataset" ]] || mp1_fail "Dataset is not readable: $mp1_dataset"
mp1_run_root="${MP1_RUN_ROOT:-$mp1_project_root/runs}"
[[ "$mp1_run_root" == /* ]] || mp1_run_root="$mp1_project_root/$mp1_run_root"

mp1_all_groups=(MP1_baseline MP1_mgad MP1_diag MP1_full MP1_mgad_diag MP1_mgad_full)
if [[ "$mp1_requested_group" == all ]]; then
  mp1_groups=("${mp1_all_groups[@]}")
else
  mp1_known_group=false
  for mp1_group in "${mp1_all_groups[@]}"; do
    [[ "$mp1_requested_group" != "$mp1_group" ]] || mp1_known_group=true
  done
  [[ "$mp1_known_group" == true ]] || mp1_fail "Unknown group: $mp1_requested_group"
  mp1_groups=("$mp1_requested_group")
fi

# Preflight every selected output before starting the first serial run.
for mp1_group in "${mp1_groups[@]}"; do
  mp1_run_name="${mp1_group}_${mp1_profile}_${mp1_task}_demo${mp1_n_demo}_s${mp1_seed}"
  mp1_output="$mp1_run_root/$mp1_run_name"
  if [[ "$mp1_resume" == true ]]; then
    [[ -f "$mp1_output/checkpoints/latest.ckpt" ]] || mp1_fail "Resume checkpoint is missing: $mp1_output/checkpoints/latest.ckpt"
  elif [[ -e "$mp1_output" && ( ! -d "$mp1_output" || -n "$(find "$mp1_output" -mindepth 1 -maxdepth 1 -print -quit)" ) ]]; then
    mp1_fail "New run would reuse an existing nonempty output: $mp1_output. Choose another MP1_RUN_ROOT or explicitly resume."
  fi
done

cd -- "$mp1_project_root"
export CUDA_VISIBLE_DEVICES="$mp1_gpu"
printf 'SOURCE=%s\nPROJECT=%s\nDATASET=%s\nPROFILE=%s\nGPU=%s\n' \
  "$mp1_source_root" "$mp1_project_root" "$mp1_dataset" "$mp1_profile" "$mp1_gpu"
printf 'EPOCHS=%s\nROLLOUT_EVERY=%s\nCHECKPOINT_EVERY=%s\n' \
  "${mp1_epochs:-CONFIG_DEFAULT: int(50000/n_demo)}" \
  "${mp1_rollout_every:-CONFIG_DEFAULT: int(1000/n_demo)}" \
  "${mp1_checkpoint_every:-CONFIG_DEFAULT: int(1000/n_demo)}"

mp1_budget_overrides=()
[[ -z "$mp1_epochs" ]] || mp1_budget_overrides+=("training.num_epochs=$mp1_epochs")
[[ -z "$mp1_rollout_every" ]] || mp1_budget_overrides+=("training.rollout_every=$mp1_rollout_every")
[[ -z "$mp1_checkpoint_every" ]] || mp1_budget_overrides+=("training.checkpoint_every=$mp1_checkpoint_every")

for mp1_group in "${mp1_groups[@]}"; do
  mp1_run_name="${mp1_group}_${mp1_profile}_${mp1_task}_demo${mp1_n_demo}_s${mp1_seed}"
  mp1_output="$mp1_run_root/$mp1_run_name"
  mp1_command=("$mp1_python" -u "$mp1_entry" --config-name "$mp1_group"
    "profile=$mp1_profile" "task_name=$mp1_task" "n_demo=$mp1_n_demo"
    "dataset_path=$mp1_dataset" "task.env_runner.dataset_path=$mp1_dataset"
    "training.seed=$mp1_seed"
    "training.device=cuda:0" "training.resume=$mp1_resume"
    "training.enable_rollouts=true" "training.enable_validation=true"
    "training.max_train_steps=$mp1_max_train_steps" "training.max_val_steps=$mp1_max_val_steps"
    "dataloader.batch_size=$mp1_batch_size" "val_dataloader.batch_size=$mp1_batch_size"
    "dataloader.num_workers=$mp1_workers" "val_dataloader.num_workers=$mp1_workers"
    "dataloader.persistent_workers=false" "val_dataloader.persistent_workers=false"
    "task.env_runner.n_envs=$mp1_n_envs" "task.env_runner.n_train=$mp1_n_train"
    "task.env_runner.n_test=$mp1_n_test" "task.env_runner.n_train_vis=$mp1_n_train_vis"
    "task.env_runner.n_test_vis=$mp1_n_test_vis" "checkpoint.save_ckpt=true"
    "logging.backend=swanlab" "logging.mode=online" "logging.project=$mp1_logging_project"
    "logging.name=$mp1_run_name" "hydra.run.dir=$mp1_output")
  mp1_command+=("${mp1_budget_overrides[@]}")
  printf '\nRUN %s\n' "$mp1_run_name"
  printf '%q ' "${mp1_command[@]}"
  printf '\n'
  if [[ "$mp1_dry_run" == true ]]; then
    continue
  fi
  # Verification does not load the real data; the path checks above are separate.
  "${mp1_command[@]}" --verify
  mkdir -p -- "$mp1_output"
  "${mp1_command[@]}" 2>&1 | tee -a "$mp1_output/console.log"
done
