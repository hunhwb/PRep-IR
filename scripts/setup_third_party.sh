#!/usr/bin/env bash
# Clone the external code this repository calls, at the commits the paper used, into third_party/:
#   DiffReg-PBIR  (2b1673a)  surface reconstruction, DiffusionRenderer G-buffers, Stanford-ORB / DTC evaluation scripts
#   MVInverse     (6172ff9)  material / normal predictor
#   Stanford-ORB  (8a3bdae)  official benchmark code, with DiffReg-PBIR's RGBA-EXR patch
# Then build the environments with scripts/setup_environment.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/third_party" && cd "$ROOT/third_party"
clone() {   # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
}
clone https://github.com/gerwang/DiffReg-PBIR.git DiffReg-PBIR 2b1673a
git -C DiffReg-PBIR submodule update --init --recursive
clone https://github.com/Maddog241/mvinverse.git mvinverse 6172ff9
clone https://github.com/StanfordORB/Stanford-ORB.git Stanford-ORB 8a3bdae
git -C Stanford-ORB apply --check "$ROOT/third_party/DiffReg-PBIR/scripts/stanford_orb/rgba_exr.patch" 2>/dev/null \
  && git -C Stanford-ORB apply "$ROOT/third_party/DiffReg-PBIR/scripts/stanford_orb/rgba_exr.patch"
echo "third_party ready: $(ls)"
