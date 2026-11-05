#!/usr/bin/env bash
# Build the from-source parts of the environments, after `pixi install` and scripts/setup_third_party.sh. Idempotent.
#   (default)  this repository's environment (pixi.toml): mmcv 1.7.0, tiny-cuda-nn, pytorch3d, DiffReg-PBIR's
#              neural_surface_recon CUDA kernels, DiffReg-PBIR's Mitsuba fork (scalar_rgb, cuda_ad_rgb; built in
#              third_party/DiffReg-PBIR/diffusion_reg/deps/mitsuba3/build) and Blender 3.6
#   cosmos     envs/cosmos-predict1 (DiffusionRenderer prior only): cosmos1-diffusion-renderer's requirements.txt and
#              TransformerEngine v1.12
# The build steps follow DiffReg-PBIR's README (Setup, steps 2-3). CUDA architectures come from TORCH_CUDA_ARCH_LIST
# (e.g. "8.9" or "8.9;12.0"), by default the compute capability of the first visible GPU, so a machine without a GPU
# can build for another one.
#
# usage: bash scripts/setup_environment.sh [cosmos]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
DIFFREG=$(realpath "${PREP_IR_DIFFREG:-$ROOT/third_party/DiffReg-PBIR}")
ARCHS=${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)}
[ -n "$ARCHS" ] || { echo "no GPU visible: set TORCH_CUDA_ARCH_LIST, e.g. 8.9" >&2; exit 1; }
export MAX_JOBS=${MAX_JOBS:-16}
MANIFEST=$ROOT/pixi.toml

build() {   # compile inside a pixi environment with its own nvcc / gcc 11, linking against the libcuda stub
  pixi run --manifest-path "$MANIFEST" env TORCH_CUDA_ARCH_LIST="$ARCHS" TCNN_CUDA_ARCHITECTURES="${ARCHS//./}" \
    NVTE_CUDA_ARCHS="${ARCHS//./}" FORCE_CUDA=1 bash -c 'set -euo pipefail; unset CFLAGS CXXFLAGS NVCC_FLAGS CUDAFLAGS
      export CUDA_HOME=$CONDA_PREFIX LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
      export CPATH=$CONDA_PREFIX/include${CPATH:+:$CPATH}; '"$1"
}
has() {   # installed according to pip metadata (importing needs a GPU for some of them)
  pixi run --manifest-path "$MANIFEST" python -m pip show -q "$1" >/dev/null 2>&1
}

if [ "${1:-}" = cosmos ]; then
  MANIFEST=$ROOT/envs/cosmos-predict1/pixi.toml
  COSMOS=$DIFFREG/diffusion_reg/deps/cosmos1-diffusion-renderer
  pixi install --manifest-path "$MANIFEST"
  has megatron-core || build "pip install -r $COSMOS/requirements.txt"
  if ! has transformer_engine; then
    # TransformerEngine compiles against the cuDNN / NVTX headers that only torch's nvidia-* wheels ship
    [ -d third_party/TransformerEngine ] || git clone --branch v1.12 --recursive https://github.com/NVIDIA/TransformerEngine.git third_party/TransformerEngine
    build 'NV=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || python -c "import nvidia; print(nvidia.__path__[0])")
      export CUDNN_PATH=$NV/cudnn; for i in $NV/*/include; do CPATH=$i:$CPATH; done
      for l in $NV/*/lib; do LIBRARY_PATH=$l:$LIBRARY_PATH; LD_LIBRARY_PATH=$l:${LD_LIBRARY_PATH:-}; done
      export CPATH LIBRARY_PATH LD_LIBRARY_PATH; [ -e $CUDNN_PATH/lib/libcudnn.so ] || ln -s libcudnn.so.9 $CUDNN_PATH/lib/libcudnn.so
      cd third_party/TransformerEngine && NVTE_FRAMEWORK=pytorch pip install --no-build-isolation -v .'
  fi
  echo "cosmos-predict1 ready ($ARCHS)"
  exit 0
fi

pixi install --manifest-path "$MANIFEST"
has mmcv || build 'pip install --no-build-isolation mmcv==1.7.0'
has tinycudann || build 'pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"'
has pytorch3d || build 'pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"'
has neural_pbir_cuda_utils || build "cd $DIFFREG/neural_surface_recon/lib/cuda && pip install --no-build-isolation . && rm -rf build"
B=$DIFFREG/diffusion_reg/deps/mitsuba3/build
if [ ! -f "$B/setpath.sh" ] || [ ! -d "$B/python/mitsuba" ]; then
  # the environment's zlib must be explicit and /usr ignored, or the system zlib header breaks libpng against the sysroot;
  # Dr.Jit's Python extension first, as DiffReg-PBIR's README does, to avoid a race with the stub generation
  build "cmake -S $DIFFREG/diffusion_reg/deps/mitsuba3 -B $B -GNinja -DMI_DEFAULT_VARIANTS='scalar_rgb,cuda_ad_rgb' \
    -DPython_EXECUTABLE=\$CONDA_PREFIX/bin/python -DZLIB_ROOT=\$CONDA_PREFIX -DCMAKE_IGNORE_PREFIX_PATH=/usr \
    -DCMAKE_C_COMPILER=\$CC -DCMAKE_CXX_COMPILER=\$CXX && ninja -C $B drjit-python && ninja -C $B"
fi
bash "$DIFFREG/scripts/setup_blender.sh"
echo "prep-ir ready ($ARCHS); Mitsuba fork: $B"
