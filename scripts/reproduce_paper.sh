#!/usr/bin/env bash
# The paper's experiments, one scripts/run_scene.sh call per (scene, configuration). Groups:
#   main         Tab. "Synthetic benchmarks": Synthetic4Relight + TensoIR-Synthetic, DiffusionRenderer, fixed-zero and
#                learned metallic
#   orb          Tab. "Stanford-ORB": the official light split (7 captures), DiffusionRenderer, fixed-zero and learned metallic
#   dtc          Tab. "DTC-Synthetic": 7 scenes, DiffusionRenderer
#   prior_use    Tab. "How the prior enters material optimization": DiffusionRenderer with free per-vertex materials and
#                with per-vertex materials + prior loss, on S4R / TensoIR / ORB light (ours: groups main / orb)
#   mvinverse    supplement, "Results with MVInverse Priors": groups main, orb, dtc and prior_use with MVInverse
#   metallic     supplement, "Metallic Estimation": learned metallic, binarization loss (.01, .1) and dual hypothesis,
#                both priors, on S4R / TensoIR / ORB light; learned metallic on DTC
#   orb_full     supplement, "Full Stanford-ORB Evaluation": all 42 captures, both priors, fixed-zero / learned / dual
#   material_ls  supplement, "Effect of material Large Steps": lambda = 0, both priors, S4R / TensoIR / ORB light, one seed
#   geometry     supplement, "Contribution of Geometry Optimization": fixed mesh, both priors on S4R / TensoIR (seeds
#                0 1 2), DiffusionRenderer on ORB light and DTC
#   refinement   supplement, "Matched refinement comparison": unrefined priors, both priors, S4R / TensoIR / ORB light,
#                one seed
# Not scripted here: the lattice oracle fits, the update-consistency measurement, the prior-quality sweep and the
# mesh Large Steps / STAdam smoothing controls.
# SEEDS: optimization seeds (default: the paper's three seeds). Priors and meshes are shared by all seeds and
# configurations; one-seed groups use the first seed.
# usage: scripts/reproduce_paper.sh GROUP [GROUP ...];  then: python scripts/summarize.py
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEEDS=${SEEDS:-"20260916 20261916 20262916"}
X=$ROOT/configs/experiments
S4R="air_baloons chair hotdog jugs"
TIR="armadillo ficus hotdog lego"
ORB_LIGHT="teapot_scene001 grogu_scene002 gnome_scene003 car_scene004 pitcher_scene005 blocks_scene006 cactus_scene007"
ORB_FULL=$(python -c "import sys; sys.path.insert(0, '$ROOT'); from prep_ir.datasets.orb import FULL; print(' '.join(FULL))")
DTC=$(python -c "import sys; sys.path.insert(0, '$ROOT'); from prep_ir.datasets.dtc import SCENES; print(' '.join(SCENES))")

run() {   # dataset "scenes" prior [config files...] [-- run_scene.sh options]
  local ds=$1 scenes=$2 prior=$3; shift 3
  local configs=() extra=()
  while [ $# -gt 0 ]; do [ "$1" = -- ] && { shift; extra=("$@"); break; }; configs+=("$X/$1.yaml"); shift; done
  for seed in $SEEDS; do
    for s in $scenes; do
      local args=(); [ ${#configs[@]} -gt 0 ] && args+=(--config "${configs[@]}")
      [ "$seed" != 20260916 ] && args+=(--set "seed=$seed")
      "$ROOT/scripts/run_scene.sh" "$ds" "$s" "$prior" "${extra[@]}" "${args[@]}"
    done
  done
}

one() { SEEDS=${SEEDS%% *} run "$@"; }   # the first seed only
benchmarks() {   # prior: the main-paper benchmark runs of one prior
  local p=$1
  run s4r "$S4R" $p; run tir "$TIR" $p; run orb "$ORB_LIGHT" $p; run dtc "$DTC" $p
  run s4r "$S4R" $p learned_metallic; run tir "$TIR" $p learned_metallic; run orb "$ORB_LIGHT" $p learned_metallic
  for c in vertex_free vertex_prior_loss; do run s4r "$S4R" $p $c; run tir "$TIR" $p $c; run orb "$ORB_LIGHT" $p $c; done
}
DR=diffusion_renderer MVI=mvinverse

for group in "$@"; do
  case $group in
    main)        for c in "" learned_metallic; do run s4r "$S4R" $DR $c; run tir "$TIR" $DR $c; done;;
    orb)         for c in "" learned_metallic; do run orb "$ORB_LIGHT" $DR $c; done;;
    dtc)         run dtc "$DTC" $DR;;
    prior_use)   for c in vertex_free vertex_prior_loss; do run s4r "$S4R" $DR $c; run tir "$TIR" $DR $c; run orb "$ORB_LIGHT" $DR $c; done;;
    mvinverse)   benchmarks $MVI;;
    metallic)    for p in $DR $MVI; do
                   for c in learned_metallic metallic_binarization_001 metallic_binarization_01 dual_hypothesis; do
                     run s4r "$S4R" $p $c; run tir "$TIR" $p $c; run orb "$ORB_LIGHT" $p $c
                   done
                   run dtc "$DTC" $p learned_metallic
                 done;;
    orb_full)    for p in $DR $MVI; do for c in "" learned_metallic dual_hypothesis; do run orb "$ORB_FULL" $p $c; done; done;;
    material_ls) for p in $DR $MVI; do one s4r "$S4R" $p no_material_ls; one tir "$TIR" $p no_material_ls; one orb "$ORB_LIGHT" $p no_material_ls; done;;
    geometry)    for p in $DR $MVI; do SEEDS="0 1 2" run s4r "$S4R" $p fixed_geometry; SEEDS="0 1 2" run tir "$TIR" $p fixed_geometry; done
                 run orb "$ORB_LIGHT" $DR fixed_geometry; run dtc "$DTC" $DR fixed_geometry;;
    refinement)  for p in $DR $MVI; do for ds in s4r tir orb; do
                   case $ds in s4r) sc=$S4R;; tir) sc=$TIR;; orb) sc=$ORB_LIGHT;; esac
                   one $ds "$sc" $p -- --no-refine-albedo
                 done; done;;
    *) echo "unknown group $group" >&2; exit 1;;
  esac
done
