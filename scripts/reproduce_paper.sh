#!/usr/bin/env bash
# The paper's experiments, one scripts/run_scene.sh call per (scene, configuration). Groups:
# All groups use the default albedo refinement (spatial sigma 12, prior-colour sigma 0.2),
# including the per-vertex controls. The shared initial mesh and predictions stay fixed.
#   main          "Synthetic benchmarks": Synthetic4Relight + TensoIR-Synthetic, both priors
#   orb           "Stanford-ORB": the official light split (7 captures), both priors
#   dtc           "DTC-Synthetic relighting": 7 scenes, both priors
#   prior_use     "How the prior enters the material optimization": MVInverse with free per-vertex materials and with
#                 per-vertex materials + prior loss, on S4R / TensoIR / ORB light (ours: group main / orb)
#   speedup       "Optimization in a minute": the 200-step schedule on Synthetic4Relight for ours (both priors) and the
#                 two per-vertex controls
#   material_ls   supplement, "Effect of material Large Steps": lambda = 0, both priors, S4R / TensoIR / ORB light
#   metallic      supplement, "Learned Metallic Setting": both priors, S4R / TensoIR / DTC / ORB full
#   orb_full      supplement, "Full Stanford-ORB Evaluation": all 42 captures, both priors
# SEEDS: optimization seeds (default: the paper's single seed 20260916; the four-seed protocol used
# "20260916 20261916 20262916 20263916"). Priors and meshes are shared by all seeds and configurations.
# usage: scripts/reproduce_paper.sh GROUP [GROUP ...];  then: python scripts/summarize.py
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEEDS=${SEEDS:-20260916}
X=$ROOT/configs/experiments
S4R="air_baloons chair hotdog jugs"
TIR="armadillo ficus hotdog lego"
ORB_LIGHT="teapot_scene001 grogu_scene002 gnome_scene003 car_scene004 pitcher_scene005 blocks_scene006 cactus_scene007"
ORB_FULL=$(python -c "import sys; sys.path.insert(0, '$ROOT'); from prep_ir.datasets.orb import FULL; print(' '.join(FULL))")
DTC=$(python -c "import sys; sys.path.insert(0, '$ROOT'); from prep_ir.datasets.dtc import SCENES; print(' '.join(SCENES))")

run() {   # dataset "scenes" prior [config files...]
  local ds=$1 scenes=$2 prior=$3; shift 3
  local configs=(); for c in "$@"; do configs+=("$X/$c.yaml"); done
  for seed in $SEEDS; do
    for s in $scenes; do
      local args=(); [ ${#configs[@]} -gt 0 ] && args+=(--config "${configs[@]}")
      [ "$seed" != 20260916 ] && args+=(--set "seed=$seed")
      "$ROOT/scripts/run_scene.sh" "$ds" "$s" "$prior" "${args[@]}"
    done
  done
}

for group in "$@"; do
  case $group in
    main)        for p in mvinverse diffusion_renderer; do run s4r "$S4R" $p; run tir "$TIR" $p; done;;
    orb)         for p in mvinverse diffusion_renderer; do run orb "$ORB_LIGHT" $p; done;;
    dtc)         for p in mvinverse diffusion_renderer; do run dtc "$DTC" $p; done;;
    prior_use)   for c in vertex_free vertex_prior_loss; do run s4r "$S4R" mvinverse $c; run tir "$TIR" mvinverse $c; run orb "$ORB_LIGHT" mvinverse $c; done;;
    speedup)     run s4r "$S4R" mvinverse fast200; run s4r "$S4R" diffusion_renderer fast200
                 run s4r "$S4R" mvinverse vertex_free fast200; run s4r "$S4R" mvinverse vertex_prior_loss fast200;;
    material_ls) for p in mvinverse diffusion_renderer; do run s4r "$S4R" $p no_material_ls; run tir "$TIR" $p no_material_ls; run orb "$ORB_LIGHT" $p no_material_ls; done;;
    metallic)    for p in mvinverse diffusion_renderer; do
                   run s4r "$S4R" $p learned_metallic; run tir "$TIR" $p learned_metallic; run dtc "$DTC" $p learned_metallic; run orb "$ORB_FULL" $p learned_metallic
                 done;;
    orb_full)    for p in mvinverse diffusion_renderer; do run orb "$ORB_FULL" $p; done;;
    *) echo "unknown group $group" >&2; exit 1;;
  esac
done
