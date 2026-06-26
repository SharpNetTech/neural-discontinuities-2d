<div align="center">
<h1>SharpNet: Enhancing MLPs to Represent Functions with Controlled Non&#8288;-&#8288;differentiability</h1>

[ACM DL](https://doi.org/10.1145/3811330) | [Homepage](https://sharpnettech.github.io) | [SharpNet2D](https://github.com/SharpNetTech/SharpNet2D) | Coming soon...

ACM Transactions on Graphics (SIGGRAPH 2026)

<p><span><b>Hanting Niu</b><sup>1,2,*</sup></span> · <span><b>Junkai Deng</b><sup>3,*</sup></span> · <span><b>Fei Hou</b><sup>1,2</sup></span> · <span><b>Wencheng Wang</b><sup>1,2</sup></span> · <span><b>Ying He</b><sup>3</sup></span></p>
<p><sup>1</sup> Institute of Software, Chinese Academy of Sciences<br>
<sup>2</sup> University of Chinese Academy of Sciences<br>
<sup>3</sup> Nanyang Technological University</p>
<p><sup>*</sup> Equal contributions</p>
</div>

## Neural Discontinuities 2D ##

This is a supplemental code release that modifies the official code release of paper "2D Neural Fields with Learned Discontinuities" by Liu et al. (Eurographics 2025). The modified code is designed to run an experiment introduced in Section 4.3 of our paper.

## What does this repo do? ##
This repo should be able to reproduce the following experiments:
| | Geodesic<br>(Section 4.1) | Medial axis<br>(Section 4.2) | Belhe<br>(Section 4.3) |
|--|:--:|:----:|:----:|
| Raw MLP | ✗ | - | ✗ |
| InstantNGP | ✗ | - | ✗ |
| SharpNet w/ ReLU | ✗ | - | ✗ |
| SharpNet w/ Softplus (Ours) | ✗ | ✗ | ✗ |
| Belhe et al | - | - | ✗ |
| Liu et al | - | - | ✓ |

Note: The experiment in Section 4.3 is conveniently named "Belhe" because the feature edges are taken directly from Belhe et al. It should not be confused with the actual method.

## The original readme is retained and is also a must-read ##
You can find the original README file at [README_orig.md](README_orig.md). The original readme file contains information on how to setup the environment, project, and how to run the code.

### If you have difficulties with setup ###
We provide a Dockerfile that can build a working environment. The Dockerfile is not optimized for image space but at least it works.

```sh
docker build -t liu2025:latest .
docker run --rm -it --gpus=all --ipc=host -v /path/to/code:/workspace liu2025:latest bash
```

## Our modifications to the original code ##
We did the following modifications:
* Renamed the original readme and written this readme;
* Written a `Dockerfile`;
* Modified the neural network to add a functionality of exporting 2D field;
* Modified the training pipeline to allow flexible output and hidden dimensions, as well as make the code more robust;
* Added Belhe experiment training code, Belhe experiment data generation code and mesh generation code.

## After you set up the project ##
The code by Liu et al is built heavily on the premise of fitting a discrete 2D image. This makes query of field value at arbitrary locations impossible.

You need to generate the 2D image first. Run the following code:
```sh
python inputs/belhe/make_belhe.py
```

A file `inputs/belhe/belhe.npy`, which is a 512-by-512 image, will be produced.

## Running the code ##
Our training code for the Belhe experiment can be run directly. That is,
```sh
python launching/run_belhe.py inputs/belhe/belhe.npy inputs/belhe/belhe.svg --config-json inputs/belhe/belhe.json --snapshot outputs/belhe
```

Results will be stored to `./outputs/belhe`.

Run `launching/make_mesh.py` to build the mesh.
```sh
python launching/make_mesh.py outputs/belhe/output.npy
```

A file `outputs/belhe/output.ply` will appear.

## Citation ##
If you find our work useful, please cite SharpNet.
```bibtex
@article{niu2026sharpnet,
    author = {Niu, Hanting and Deng, Junkai and Hou, Fei and Wang, Wencheng and He, Ying},
    title = {{SharpNet}: Enhancing {MLP}s to Represent Functions with Controlled Non-differentiability},
    year = {2026},
    issue_date = {July 2026},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {45},
    number = {4},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3811330},
    doi = {10.1145/3811330},
    journal = {ACM Transactions on Graphics},
    month = jul,
    articleno = {113},
    numpages = {19},
    keywords = {MLP, Sharp features, Poisson's equation, Jump Neumann boundary condition, Green's function, CAD},
}
```

## Acknowledgments ##
We thank the authors for their code. The original codebase can be found at [squidrice21/neural-discontinuities-2d](https://github.com/squidrice21/neural-discontinuities-2d).
