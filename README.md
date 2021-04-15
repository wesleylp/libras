# A Gait Energy Image-Based System for Brazilian Sign Language Recognition

This work was submitted to the [IEEE Transactions on Circuit and Systems I: Regular Papers (TCAS-I) - Special Issue on Regional Flagship Conferences of the IEEE Circuits and Systems Society ](https://ieee-cas.org/pubs/tcas1).

![overview](https://github.com/wesleylp/libras/blob/master/.figures/system_overview.png?raw=true)


## Abstract

This work, address the problem of Libras recognition in video. To do so, we employ a two-step method with feature space mapping and classification. First, we sgmente the body parts of each subject in a video through [DensePose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose) estimation model. Then, we use Gait Energy Image to encode the motion of the body parts in a compact feature space, as illustrated in the Figure below.

![GEI](https://github.com/wesleylp/libras/blob/master/.figures/generate_GEI.png?raw=true)

The pipeline used in the classification step is illustrated in the Figure below. The input is the GEI representation that is cropped in the region of interest that contains movement The cropped samples are then reshaped to a smaller size, while keeping the aspect ratio of the original videos. We employ dimensionality reduction (or SMOTE) as a solution to the curse of dimensionality. After the dimensionality reduction (or data augmentation), the samples submitted to a pipeline classifier.

![cls pipeline](https://github.com/wesleylp/libras/blob/master/.figures/classification_pipeline.png?raw=true)

## Results

We perform experiments on three challenging Brazilian sign language (Libras) datasets, CEFET/RJ-Libras, MINDS-Libras, and LIBRAS-UFOP.

### CEFET/RJ-Libras

![metrics CEFET](https://github.com/wesleylp/libras/blob/master/.figures/metrics_CEFET.png?raw=true) ![cfn mtx CEFET](https://github.com/wesleylp/libras/blob/master/.figures/cfnmtx_CEFET.png?raw=true)

![boxplot CEFET](https://github.com/wesleylp/libras/blob/master/.figures/boxplot_CEFET.png?raw=true)

### MINDS-Libras

![metrics MINDS](https://github.com/wesleylp/libras/blob/master/.figures/metrics_MINDS.png?raw=true) ![cfn mtx MINDS](https://github.com/wesleylp/libras/blob/master/.figures/cfnmtx_MINDS.png?raw=true)

![boxplot MINDS](https://github.com/wesleylp/libras/blob/master/.figures/boxplot_MINDS.png?raw=true)

### LIBRAS-UFOP-ISO

![metrics UFOP](https://github.com/wesleylp/libras/blob/master/.figures/metrics_UFOP.png?raw=true) ![cfn mtx UFOP](https://github.com/wesleylp/libras/blob/master/.figures/cfnmtx_UFOP.png?raw=true)

![boxplot UFOP](https://github.com/wesleylp/libras/blob/master/.figures/boxplot_UFOP.png?raw=true)
