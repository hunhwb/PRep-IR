# Prior-Space Reparameterization for Physically Based Inverse Rendering

Code for *Integrating Learned Priors through Prior-Space Reparameterization for Physically Based Inverse Rendering*.
Recovers geometry, PBR materials and illumination from multi-view images using MVInverse or DiffusionRenderer priors
and a photometric objective.

## Installation

Requires Linux, an NVIDIA GPU and [pixi](https://pixi.sh). One environment (`pixi.toml`, locked in `pixi.lock`;
Python 3.11, CUDA 12.8 PyTorch, Mitsuba 3.9.1) runs every step except DiffusionRenderer inference, including
DiffReg-PBIR's surface reconstruction and the Stanford-ORB / DTC evaluations.

```bash
bash scripts/setup_third_party.sh   # DiffReg-PBIR, MVInverse and Stanford-ORB at the commits the paper used
bash scripts/setup_environment.sh   # pixi install + the CUDA extensions, DiffReg-PBIR's Mitsuba fork and Blender 3.6
pixi run test
pixi shell                          # the commands below run inside this environment
```

`setup_environment.sh` compiles for the visible GPU; set `TORCH_CUDA_ARCH_LIST` (e.g. `8.9;12.0`) to build on a machine
without one. MVInverse downloads its weights on first use.

The DiffusionRenderer prior needs a second environment, `envs/cosmos-predict1`, and the Cosmos checkpoints (see
DiffReg-PBIR's README_DATA, "Step A"):

```bash
bash scripts/setup_environment.sh cosmos
```

External locations can be overridden with `PREP_IR_DIFFREG`, `PREP_IR_MVINVERSE`, `PREP_IR_ORB_BENCHMARK`,
`PREP_IR_MITSUBA_SETPATH`, `PREP_IR_COSMOS_ENV` (a pixi manifest) and `BLENDER`; defaults point into `third_party/` and
`envs/`.

## Data

Download the datasets below, then convert them.

| Dataset | Source | Conversion |
|---|---|---|
| Synthetic4Relight | [InvRender](https://github.com/zju3dv/InvRender) | `python scripts/prepare_dataset.py s4r --src <Synthetic4Relight> --out data/s4r` |
| TensoIR-Synthetic | [TensoIR_Material](https://github.com/junkzhu/PTIR-GS) (armadillo, ficus, hotdog) and [TensoIR](https://github.com/Haian-Jin/TensoIR) (lego and envmaps) | `python scripts/prepare_dataset.py tir --src <TensoIR_Material> --tensoir <TensoIR-Synthetic> --out data/tir` |
| Stanford-ORB | [Stanford-ORB](https://stanfordorb.github.io) (`blender_HDR`, `ground_truth`) | `python scripts/prepare_dataset.py orb --src <stanford_orb> --out data/orb` |
| DTC-Synthetic | [DiffReg-PBIR release](https://github.com/gerwang/DiffReg-PBIR/releases/tag/dtc-synthetic-v1) (extracted scene archives) | `python scripts/prepare_dataset.py dtc --src <extracted zips> --out data/dtc` |

Converted scenes go to `data/<dataset>/<scene>/`. For all 42 ORB captures, add `--scenes full` to its conversion command.

## Running one scene

```bash
scripts/run_scene.sh s4r hotdog mvinverse
scripts/run_scene.sh s4r hotdog diffusion_renderer
```

The script runs prediction, mesh reconstruction, prior fusion, optimization and evaluation, skipping completed steps.
Albedo refinement is enabled by default: spatial sigma **12**, prior-colour sigma **0.2**, input-colour sigma **0.03**.
Lattice spacing is estimated after refinement; material Large Steps uses **20**. Use `--no-refine-albedo` to disable
refinement.

Prepared scenes are saved as `work/<dataset>/<scene>/scene_<prior>_refined.pt`; results go to
`outputs/<dataset>/<scene>/<prior>_refined/`. Results include `checkpoint.pt`, `result/result.npz`, `run.json` and
`eval/evaluation.json`.

Training settings live in `configs/default.yaml`; experiment files and `--set` provide overrides:

```bash
scripts/run_scene.sh s4r hotdog mvinverse --config configs/experiments/fast200.yaml --set seed=20261916
```

| Experiment file | Change |
|---|---|
| `fast200.yaml` | 200 optimization steps |
| `no_material_ls.yaml` | Disable material Large Steps |
| `learned_metallic.yaml` | Learn metallic, initialized at 0.2 |
| `vertex_free.yaml` | Free per-vertex material control |
| `vertex_prior_loss.yaml` | Per-vertex material control with a prior loss |

Individual pipeline steps are available under `scripts/`; use `--help` on the Python entry points.

## Reproducing the paper

```bash
scripts/reproduce_paper.sh main orb dtc prior_use speedup
scripts/reproduce_paper.sh material_ls metallic orb_full
SEEDS="20260916 20261916 20262916" scripts/reproduce_paper.sh main
python scripts/summarize.py
```

All groups use default refinement and share initial meshes and priors across optimization seeds.

## Evaluation

```bash
python scripts/evaluate.py outputs/s4r/hotdog/mvinverse_refined
# Quick synthetic check:
python scripts/evaluate.py outputs/s4r/hotdog/mvinverse_refined --view-stride 4
```

| Dataset | Protocol |
|---|---|
| S4R / TensoIR | 200 test views by default; `--view-stride 4` scores 50. A scene-level RGB albedo scale uses all 200 views and is shared with relighting. |
| Stanford-ORB | Official novel-view and novel-light evaluation via Blender and the ORB benchmark. |
| DTC-Synthetic | Released DiffReg-PBIR protocol: 3 illuminations × 20 views, 1024 spp. |

## Acknowledgements

This codebase builds on the following wonderful works. We sincerely thank their authors for sharing their research,
code, models and datasets with the community.

- [DiffReg-PBIR](https://github.com/gerwang/DiffReg-PBIR) provides the surface reconstruction, prediction staging
  and Stanford-ORB / DTC evaluation pipelines we build on, as well as the prior loss used in our control experiment.
- [Neural-PBIR](https://github.com/facebookresearch/DigitalTwinCatalog) provides the foundations for surface
  reconstruction and the spherical Gaussian-to-texture environment-map parameterization.
- [MVInverse](https://github.com/Maddog241/mvinverse) and
  [DiffusionRenderer](https://github.com/nv-tlabs/cosmos1-diffusion-renderer) provide the learned material and normal
  predictions used by our pipeline.
- [Large Steps](https://github.com/rgl-epfl/large-steps-pytorch) and
  [cholespy](https://github.com/rgl-epfl/cholespy) provide the preconditioning formulation and sparse Cholesky solver.
- The [permutohedral lattice reference implementation](https://github.com/MiguelMonteiro/permutohedral_lattice)
  informed our lattice embedding implementation.
- [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) and [Dr.Jit](https://github.com/mitsuba-renderer/drjit)
  provide the differentiable rendering and automatic differentiation infrastructure.
- [InvRender](https://github.com/zju3dv/InvRender), [TensoIR](https://github.com/Haian-Jin/TensoIR) and
  [PTIR-GS](https://github.com/junkzhu/PTIR-GS) provide the synthetic datasets, material ground truth and evaluation
  protocols used in our experiments.
- [Stanford-ORB](https://github.com/StanfordORB/Stanford-ORB) provides the real-world benchmark and official
  evaluation code.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); see [LICENSE](LICENSE).
Third-party code, models and datasets retain their respective licenses; source files that re-implement or adapt
external work name the origin and its license in their header.
