#!/usr/bin/env bash
# One scene end to end: prior prediction -> surface reconstruction -> scene preparation -> joint optimization ->
# evaluation. Finished steps are skipped, so the script can be re-run after an interruption.
#
# usage: scripts/run_scene.sh DATASET SCENE [PRIOR] [--no-refine-albedo] [--name RUN_NAME] [--config FILE ...] [--set KEY=VALUE ...]
#   DATASET  s4r | tir | orb | dtc        (data/<DATASET>/<SCENE> from scripts/prepare_dataset.py)
#   PRIOR    diffusion_renderer (default, the main paper) | mvinverse (supplement): the same model guides the
#            surface reconstruction and provides the material prior (the paper's DR-DR and MVI-MVI configurations)
#   --config / --set are passed to scripts/train.py; the run goes to outputs/<DATASET>/<SCENE>/<RUN_NAME>
#   Albedo refinement (spatial sigma 12, prior-colour sigma 0.2) is enabled by default.
#   Default RUN_NAME: PRIOR_refined, plus experiment file names and an overridden seed.
#
# Run from this repository's environment (`pixi shell`); the DiffusionRenderer prior also uses envs/cosmos-predict1.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
DATASET=$1; SCENE=$2; shift 2
PRIOR=diffusion_renderer
if [ $# -gt 0 ] && [[ $1 != --* ]]; then PRIOR=$1; shift; fi
NAME=""; CONFIGS=(); SETS=(); REFINE=(--refine-albedo); REFINE_ALBEDO=1; mode=""
for arg in "$@"; do
  case $arg in
    --name) mode=name;; --config) mode=config;; --set) mode=set;;
    --refine-albedo) REFINE=(--refine-albedo); REFINE_ALBEDO=1; mode="";;
    --no-refine-albedo) REFINE=(--no-refine-albedo); REFINE_ALBEDO=0; mode="";;
    *) case $mode in name) NAME=$arg;; config) CONFIGS+=("$arg");; set) SETS+=("$arg");; *) echo "unexpected $arg" >&2; exit 1;; esac;;
  esac
done
if [ -z "$NAME" ]; then
  NAME=$PRIOR
  [ "$REFINE_ALBEDO" = 1 ] && NAME+="_refined"
  for c in "${CONFIGS[@]}"; do NAME+="_$(basename "${c%.yaml}")"; done
  for s in "${SETS[@]}"; do [[ $s == seed=* ]] && NAME+="_seed${s#seed=}"; done
fi
DATA=$ROOT/data/$DATASET/$SCENE
WORK=$ROOT/work/$DATASET/$SCENE
RUN=$ROOT/outputs/$DATASET/$SCENE/$NAME
[ -f "$DATA/dataset.json" ] || { echo "$DATA: run scripts/prepare_dataset.py first" >&2; exit 1; }
diffreg_data=$(python -c "import json; print(json.load(open('$DATA/dataset.json'))['diffreg_data'])")

# 1. prior predictions, and the normal guidance of the surface reconstruction
case $PRIOR in
  mvinverse)
    [ -f "$WORK/priors/mvinverse/mapping.json" ] || python scripts/predict_mvinverse.py --data "$DATA" --out "$WORK/priors/mvinverse"
    NSR=$WORK/nsr_data/mvinverse/$SCENE
    [ -d "$NSR/normal_single" ] || python scripts/mvinverse_normals.py --data "$DATA" --prior "$WORK/priors/mvinverse" --out "$NSR";;
  diffusion_renderer)
    NSR=$(realpath "$DATA/$diffreg_data")
    [ -n "$(ls -A "$NSR/basecolor_single" 2>/dev/null)" ] || scripts/predict_diffusion_renderer.sh "$DATA"
    [ -f "$WORK/priors/diffusion_renderer/mapping.json" ] || python scripts/convert_diffusion_renderer.py --data "$DATA" --out "$WORK/priors/diffusion_renderer";;
  *) echo "unknown prior $PRIOR" >&2; exit 1;;
esac

# 2. surface reconstruction (to_worlds_after.txt is its last output)
[ -f "$WORK/mesh/$PRIOR/to_worlds_after.txt" ] || python scripts/reconstruct_mesh.py --data "$DATA" --nsr-data "$NSR" --out "$WORK/mesh/$PRIOR"

# 3. scene preparation: keep refined and original priors in separate bundles
SCENE_FILE="$WORK/scene_$PRIOR.pt"
[ "$REFINE_ALBEDO" = 1 ] && SCENE_FILE="$WORK/scene_${PRIOR}_refined.pt"
[ -f "$SCENE_FILE" ] || python scripts/prepare_scene.py --data "$DATA" --mesh "$WORK/mesh/$PRIOR/mesh.obj" --prior "$WORK/priors/$PRIOR" --out "$SCENE_FILE" "${REFINE[@]}"

# 4. joint optimization
train_args=()
[ ${#CONFIGS[@]} -gt 0 ] && train_args+=(--config "${CONFIGS[@]}")
[ ${#SETS[@]} -gt 0 ] && train_args+=(--set "${SETS[@]}")
[ -f "$RUN/result/result.npz" ] || python scripts/train.py --scene "$SCENE_FILE" --out "$RUN" "${train_args[@]}"

# 5. evaluation
[ -f "$RUN/eval/evaluation.json" ] || python scripts/evaluate.py "$RUN"
echo "$RUN/eval/evaluation.json"
