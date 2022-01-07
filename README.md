# A Gait Energy Image-Based System for Brazilian Sign Language Recognition

- [Overview](#overview)
- [Databases](#databases)
- [Results](#results)
  - [CEFET/RJ-Libras](#results-cefet)
  - [MINDS-Libras](#results-minds)
  - [LIBRAS-UFOP-ISO](#results-ufop)
- [Citation](#citation)

<a name="overview"></a>

## Overview

This work addresses the problem of Libras recognition in video. The overview of the proposed system is shown in the Figure below.

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/system_overview.png?raw=true" align="center"/></p>

The system employs a two-step method with feature space mapping and classification. First, we segment the body parts of each subject in a video through [DensePose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose) estimation model. Then, we use Gait Energy Image (GEI) to encode the motion of the body parts in a compact feature space, as illustrated in the Figure below.

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/generate_GEI.png?raw=true" align="center" width="600"/></p>

The pipeline used in the classification step is illustrated in the Figure below. The input is the GEI representation that is cropped in the region of interest that contains movement. The cropped samples are then reshaped to a smaller size, while keeping the aspect ratio of the original video frames. We employ dimensionality reduction (or SMOTE) as a solution to the curse of dimensionality. After the dimensionality reduction (or data augmentation), the samples are submitted to a classification pipeline.

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/classification_pipeline.png?raw=true" align="center" width="400"/></p>


<a name="databases"></a>

## Databases

- **CEFET/RJ-Libras**: available under request. (Please, see the corresponding author in the original database [paper](https://doi.org/10.1109/LARS/SBR/WRE51543.2020.9307017)).
- **MINDS-Libras**: available through this [link](https://zenodo.org/record/4322984#.YcEjDXXMKV4). Please, see also their  [paper](https://link.springer.com/article/10.1007/s00521-021-05802-4).
- **LIBRAS-UFOP-ISO**: also available under request through this [link](https://bit.ly/39TYl7V) ([paper](https://doi.org/10.1016/j.jvcir.2020.102772)).

<a name="results"></a>

## Results

We perform experiments on three challenging Brazilian sign language (Libras) datasets, CEFET/RJ-Libras, MINDS-Libras, and LIBRAS-UFOP.

<a name="results-cefet"></a>
<details>
<summary>CEFET/RJ-Libras</summary>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/metrics_CEFET.png?raw=true" align="center" width="300"/></p>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/cfnmtx_CEFET.png?raw=true" align="center" width="400"/></p>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/boxplot_CEFET.png?raw=true" align="center" width="300"/></p>

</details>

<a name="results-minds"></a>
<details>
<summary>MINDS-Libras</summary>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/metrics_MINDS.png?raw=true" align="center" width="300"/></p>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/cfnmtx_MINDS.png?raw=true" align="center" width="400"/></p>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/boxplot_MINDS.png?raw=true" align="center" width="300"/></p>

</details>

<a name="results-ufop"></a>
<details>
<summary> LIBRAS-UFOP-ISO </summary>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/metrics_UFOP.png?raw=true" align="center" width="400"/></p>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/cfnmtx_UFOP.png?raw=true" align="center" width="400"/></p>

<p align="center">
<img src="https://github.com/wesleylp/libras/blob/master/.figures/boxplot_UFOP.png?raw=true" align="center" width="400"/></p>

</details>

<a name="citation"></a>

## Citation

This work was published in the [IEEE Transactions on Circuit and Systems I: Regular Papers (TCAS-I) - Special Issue on Regional Flagship Conferences of the IEEE Circuits and Systems Society ](https://ieeexplore.ieee.org/document/9466162).

If you use this code for your research, please consider citing:

```
@article{passos2021,
  author={Passos, Wesley L. and Araujo, Gabriel M. and Gois, Jonathan N. and de Lima, Amaro A.},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers},
  title={A Gait Energy Image-Based System for Brazilian Sign Language Recognition},
  year={2021},
  volume={68},
  number={11},
  pages={4761-4771},
  doi={10.1109/TCSI.2021.3091001}}
```


```
