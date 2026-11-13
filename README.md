# Prior-Space Reparameterization for Physically Based Inverse Rendering

Code for *Integrating Learned Priors through **P**rior-Space **Rep**arameterization for Physically Based **I**nverse **R**endering*.
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
scripts/run_scene.sh s4r hotdog              # DiffusionRenderer prior (default, the main paper)
scripts/run_scene.sh s4r hotdog mvinverse    # MVInverse prior (supplement)
```

The script runs prediction, mesh reconstruction, prior fusion, optimization and evaluation, skipping completed steps.
Albedo refinement is enabled by default. 
Use `--no-refine-albedo` to disable refinement. We recommend enabling it for better visual quality.

Prepared scenes are saved as `work/<dataset>/<scene>/scene_<prior>_refined.pt`; results go to
`outputs/<dataset>/<scene>/<prior>_refined/`. Results include `checkpoint.pt`, `result/result.npz`, `run.json` and
`eval/evaluation.json`.

Training settings live in `configs/default.yaml`; experiment files and `--set` provide overrides:

```bash
scripts/run_scene.sh s4r hotdog --config configs/experiments/dual_hypothesis.yaml --set seed=42
```

| Experiment file | Change |
|---|---|
| `vertex_free.yaml` | Free per-vertex material control |
| `vertex_prior_loss.yaml` | Per-vertex material control with a prior loss |
| `learned_metallic.yaml` | Learn metallic, initialized at 0.2 |
| `metallic_binarization_001.yaml`, `metallic_binarization_01.yaml` | Learned metallic plus a binarization loss (weight .01 / .1) |
| `dual_hypothesis.yaml` | Dielectric–metallic dual hypothesis: binary metallic selected from photometric residuals |
| `no_material_ls.yaml` | Disable material Large Steps |
| `fixed_geometry.yaml` | Hold the reconstructed mesh fixed |
| `fast200.yaml` | Shortened 200-step schedule: about a minute of optimization per scene on one L40 |

### Dual hypothesis

With `dual_hypothesis.yaml`, every vertex carries a dielectric hypothesis (metallic 0) and a metal hypothesis
(metallic 1), each with its own base color and roughness on the same lattice. Each training view is rendered with the
selected hypothesis of every vertex, which updates geometry, illumination and the selected fields, and with the
unselected hypotheses, which updates only those fields. The per-pixel residuals of both renders are pooled to the
lattice, smoothed with the material Large Steps operator, and compared; metal is selected where it explains the images
better by a normalized margin of 0.02. The second render runs every step up to step 400 and every fourth step after,
so a run takes about 1.3–1.4 times as long as the fixed-metallic setting. `result/result.npz` holds the selected binary
material, and `result/dual.npz` both hypotheses and the selection evidence. Settings are in the `dual` section of
`configs/default.yaml`.

Individual pipeline steps are available under `scripts/`; use `--help` on the Python entry points.

## Reproducing the paper

```bash
scripts/reproduce_paper.sh main orb dtc prior_use                       # main paper (DiffusionRenderer)
scripts/reproduce_paper.sh mvinverse metallic orb_full material_ls geometry refinement   # supplement
SEEDS=20260916 scripts/reproduce_paper.sh main                          # a single seed
python scripts/summarize.py
```

Groups run the paper's three optimization seeds (20260916, 20261916, 20262916) unless `SEEDS` says otherwise; the
header of `scripts/reproduce_paper.sh` maps each group to its table. All groups use the default refinement except
`refinement`, and share initial meshes and priors across seeds and configurations.

## Evaluation

```bash
python scripts/evaluate.py outputs/s4r/hotdog/diffusion_renderer_refined
# Quick synthetic check:
python scripts/evaluate.py outputs/s4r/hotdog/diffusion_renderer_refined --view-stride 4
```

| Dataset | Protocol |
|---|---|
| S4R / TensoIR | 200 test views by default. set --view-stride > 1 for faster evaluation. --view-stride 4 is usually within 0.1 dB of the original numbers. |
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
