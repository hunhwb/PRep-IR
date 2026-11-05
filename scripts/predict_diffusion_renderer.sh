#!/usr/bin/env bash
# DiffusionRenderer G-buffers (Liang et al., "DiffusionRenderer: Neural Inverse and Forward Rendering with Video
# Diffusion Models"; the Cosmos inverse renderer 7B as packaged by DiffReg-PBIR) for one scene, written into its
# DiffReg-PBIR data directory as {normal,basecolor,roughness,metallic}_single/ (the layout DiffReg-PBIR's README_DATA
# describes):
#   1. stage the training views (DiffReg-PBIR scripts/preprocess/export_train_images.py)
#   2. Cosmos inverse rendering (envs/cosmos-predict1; TensoIR: all passes at once, others: normal, then the material
#      passes)
#   3. assemble the *_single maps (scripts/preprocess/dtc_diffrender_single.py)
# Steps 1 and 3 run in this repository's environment, step 2 in the Cosmos one (scripts/setup_environment.sh cosmos).
# Locations (override with environment variables):
#   PREP_IR_DIFFREG (third_party/DiffReg-PBIR), PREP_IR_COSMOS_ENV (envs/cosmos-predict1, a pixi manifest)
# usage: scripts/predict_diffusion_renderer.sh DATA_DIR      e.g. data/s4r/hotdog
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIFFREG=$(realpath "${PREP_IR_DIFFREG:-$ROOT/third_party/DiffReg-PBIR}")
COSMOS=${PREP_IR_COSMOS_ENV:-$ROOT/envs/cosmos-predict1}
DATA=$(realpath "$1")
read -r DATASET SCENE NSR <<< "$(python3 -c "import json,sys; d=json.load(open('$DATA/dataset.json')); print(d['dataset'], d['scene'], d['diffreg_data'])")"
D=$(realpath "$DATA/$NSR")                        # the DiffReg-PBIR data directory, named exactly <scene>
WORK=$(dirname "$D")/_gbuffer/$SCENE; GIN=$WORK/in/$SCENE; GOUT=$WORK/out
mkdir -p "$WORK"
IRTK="$DIFFREG:$DIFFREG/diffusion_reg/deps/irtk"
COS=$DIFFREG/diffusion_reg/deps/cosmos1-diffusion-renderer

if [ -z "$(ls -A "$GIN" 2>/dev/null)" ]; then
  (cd "$DIFFREG" && PYTHONPATH=$IRTK CUDA_VISIBLE_DEVICES= python -m scripts.preprocess.export_train_images --data_root "$D" --output_dir "$GIN")
fi
cosmos() {
  (cd "$COS" && pixi run --manifest-path "$COSMOS" bash -c "export CUDA_HOME=\$CONDA_PREFIX PYTHONPATH=$COS && \
     python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py --checkpoint_dir checkpoints \
     --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B --dataset_path $GIN --num_video_frames 1 --overlap_n_frames 0 \
     --group_mode image --video_save_folder $GOUT/ --clip_name_prefix $SCENE --skip_existing True --save_video False --chunk_mode all $*")
}
if [ "$DATASET" = tir ]; then
  cosmos
else
  cosmos --inference_passes normal
  cosmos --inference_passes basecolor roughness metallic
fi
(cd "$DIFFREG" && PYTHONPATH=$IRTK python -m scripts.preprocess.dtc_diffrender_single --data_root "$D" --diffrender_path "$GOUT/gbuffer_frames/$SCENE")
for k in normal basecolor roughness metallic; do
  [ -n "$(ls -A "$D/${k}_single" 2>/dev/null)" ] || { echo "no ${k}_single maps in $D" >&2; exit 1; }
done
echo "G-buffers -> $D/*_single"
