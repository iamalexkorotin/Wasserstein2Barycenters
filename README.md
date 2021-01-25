# Continuous Wasserstein-2 Barycenter Estimation without Minimax Optimization
This is the official `Python` implementation of the [ICLR 2021](https://iclr.cc/Conferences/2021) paper **Continuous Wasserstein-2 Barycenter Estimation without Minimax Optimization** (paper on [openreview](https://openreview.net/forum?id=3tFAs5E-Pe)) by [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Lingxiao Li](https://scholar.google.com/citations?user=rxQDLWcAAAAJ&hl=en), [Justin Solomon](https://scholar.google.com/citations?user=pImSVwoAAAAJ&hl=en) and [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

The repository contains reproducible `PyTorch` source code for computing **Wasserstein-2 barycenters** in high dimensions via the **non-minimax** method (proposed in the paper) by using **input convex neural networks**. Examples are provided for various toy examples and averaging image color palettes.

The code in this repository is partially based on the code for [**Wasserstein-2 Generative Networks**](https://arxiv.org/abs/1909.13082).

<p align="center"><img src="pics/barycenter.png" width="450" /></p>

## Prerequisites
The implementation is GPU-based. Single GPU (~GTX 1080 ti) is enough to run each particular experiment. Main prerequisites are:
- [pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- CUDA + CuDNN

## Repository structure
All the experiments are issued in the form of pretty self-explanatory jupyter notebooks (`notebooks/`). For convenience, the majority of the evaluation output is preserved. Auxilary source code is moved to `.py` modules (`src/`).

### Experiments
- `notebooks/CW2B_toy_experiments.ipynb.ipynb` -- **toy experiments** (in dimensions up to 256) and subset posterior aggregation;
- `notebooks/CW2B_averaging_color_palettes.ipynb` -- averaging **color palettes** of images;
### Input convex neural networks
- `src/icnn.py` -- modules for Input Convex Neural Network architectures (**DenseICNN**, **ConvICNN**);
<p align="center"><img src="https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks/blob/master/pics/icnn.png" width="450" /></p>

## Visualized Results
### Toy Experiments (2D)
Example below containts 4 initial distributions (on the left), the ground truth barycenter (in the middle) and barycenter computed by each of 4 potentials recovered by our algithm (on the right).
<p align="center"><img src="pics/ls_sr_inputs.png" width="250"/><img hspace=10 src="pics/ls_sr_bar.png" width="250"/><img hspace=10 src="pics/ls_sr_w2cb_potentials.png" width="250"/></p>

### Color Palette Averaging (3D)
Example below demonstrates barycenters of RGB (3D) color palettes of three images.

**Original images and color palettes**
<p align="center"><img src="pics/images_orig.png" width="500"/>
<p align="center"><img src="pics/color_palettes_orig.png" width="500"/>
  
**"Averaged" images and color palettes** (estimated by each of three potentials computed by our algorithm)
<p align="center"><img src="pics/images_push.png" width="500"/>
<p align="center"><img src="pics/color_palettes_push.png" width="500"/>
