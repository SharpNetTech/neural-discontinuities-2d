2D Neural Fields with Learned Discontinuities
============================================

<strong>Chenxi Liu<sup>1</sup>, Siqi Wang<sup>2</sup>, Matthew Fisher<sup>3</sup>, Deepali Aneja<sup>3</sup>, Alec Jacobson<sup>1,3</sup></strong>

<small><sup>1</sup>University of Toronto, <sup>2</sup>New York University, Courant Institute of Mathematical Sciences, <sup>3</sup>Adobe Research, San Francisco</small>

<p align="center">
	<img src="https://discontinuity2d.github.io/representative_image.jpg" width="640"/>
</p>

This repository contains the research code for the paper “[2D Neural Fields with Learned Discontinuities](https://discontinuity2d.github.io/).” It implements our discontinuous neural field and the end-to-end pipelines to fit images (and depth maps) while recovering unknown discontinuities.

Installation
------------

### Dependencies

- Linux, NVIDIA GPU recommended
- Python 3.8 (via conda), PyTorch 2.0.1 + CUDA 11.8
- torch-scatter (installed from the PyG wheel index)
- External meshing binary: TriWild

Create and activate the environment:

```bash
# From the repo root
conda env create -f environment.yml
conda activate discontinuity2d
```

Install torch-scatter manually after activating the env:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html
```

TriWild must be discoverable; set an explicit path if not on PATH:

```bash
export TRIWILD_PATH=/path/to/TriWild
```

Running Our Method
------------------

Add the repo root to PYTHONPATH (or call as a module):

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

Run the end-to-end pipeline with a PNG (or a depth grayscale image if using `--depth`). If an SVG is not specified, a default `unknown.svg` is used:

```bash
python launching/run_pipeline.py path/to/your.png --snapshot outputs/run1
```

Optionally provide an SVG explicitly to define discontinuity locations:

```bash
python launching/run_pipeline.py path/to/your.png path/to/your.svg --snapshot outputs/run1
```

Useful options (see `--help` for full list):

- `--fit` selects the representation to optimize:
	- `unknown_discontinuity` (default): infer edges while fitting
	- `discontinuity`: use known SVG discontinuities (TriWild “known” mode)
	- `per_vertex`, `per_edge`: ablations/baselines
- `--config-json`: path to a JSON with overrides merged into defaults. Meshing parameters live here (no extra CLI flags):
	- `defgrid_config.triwild_edge_r`: TriWild target edge-length ratio (e.g., 0.003–0.02)
	- To disable remeshing: set `defgrid_config.remesh_epoch` to 0 (or omit the key)
- `--depth`: treat the input as a depth target (grayscale PNG)
- `--debug`: enables intermediate logs/plots. Default runs are quiet but show tqdm progress bars

Example presets live under `inputs/` (e.g., `inputs/diffusion_curves/diffusion_curves.json`).

Results
-------

Final artifacts are saved under `--snapshot`:

- `fit_final.png`: single-sample render (spp = 1) at input resolution
- `fit_final_spp.png`: anti-aliased render (spp = 16) at input resolution
- `fit_final_spp_2x.png`: anti-aliased render (spp = 16) at 2× upscaled resolution

Repository Layout
-----------------

- `geometry/` — meshing, remeshing, and geometric utilities (TriWild interface, Canny edges)
- `learning/` — sampling strategies and data helpers
- `neural/` — neural field models and utilities
- `pipeline/` — pipeline stages used by the launcher
- `launching/run_pipeline.py` — top-level entry point
- `inputs/` — sample data and JSON configs (e.g., artistic, diffusion_curves, rendering)
- `outputs/` — example outputs and snapshots

Data
----

Sample inputs and configs are provided under `inputs/`. You can point the launcher to your own image (PNG) and optional SVG. For diffusion-generated depth map cleanup, pass `--depth` and a grayscale PNG. Artistic inputs are by Oscar Chávez (CC BY-NC-SA 2.0): [inputs/artistic/009.jpg](https://www.flickr.com/photos/chavezonico/14947922046/), [inputs/artistic/014.jpg](https://www.flickr.com/photos/chavezonico/8042646625/), [inputs/artistic/019.jpg](https://www.flickr.com/photos/chavezonico/6275757320/). Other inputs are created by us and can be used under CC0.

Troubleshooting
---------------

- If `torch-scatter` fails to install, use the PyG index matching your exact Torch/CUDA.
- If TriWild is not found, set `TRIWILD_PATH` to its binary or add it to `PATH`.
- CUDA mismatches: ensure your driver/runtime align with `pytorch-cuda=11.8` in the environment.

License
-------

This project is licensed under the Apache License, Version 2.0 — see `LICENSE` for details.

Notices: see `NOTICE` for attributions and third-party notices.

BibTeX
------

```
@article{discontinuities2d,
	title={2D Neural Fields with Learned Discontinuities},
	author={Liu, Chenxi and Wang, Siqi and Fisher, Matthew and Aneja, Deepali and Jacobson, Alec},
	booktitle={Computer Graphics Forum},
	lccn={2004233670},
	issn={0167-7055},
	year={2025},
	publisher={North Holland},
	volume={44},
	year={2025-05},
	organization={Wiley Online Library}
}
```
