# Snake Species Classification & Venom Detection

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat&logo=pytorch)
![Architecture](https://img.shields.io/badge/Model-ConvNeXt%20V2-green?style=flat)
![Optimization](https://img.shields.io/badge/Tuning-Optuna-orange?style=flat)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat&logo=nvidia)
![Conda](https://img.shields.io/badge/Conda-Environment-44A833?style=flat&logo=anaconda)

Computer Vision pipeline designed to perform **fine-grained image classification** on snake species and determine their venomous status.

It leverages the State-of-the-Art (SOTA) **ConvNeXt V2** architecture, employing transfer learning and automated hyperparameter optimization (HPO) via **Optuna** to maximize predictive performance on a challenging biological dataset.

---

## Project Overview

The objective of this project is to automate the identification of snake species from raw images. The system solves two tasks:
1.  **Multi-class Classification:** Predicting the specific species of the snake.
2.  **Binary Classification:** Inferring whether the snake is venomous based on the identified species.

This solution addresses common CV challenges such as high intra-class variance (snakes of the same species looking different) and low inter-class variance (different species looking very similar).

## Methodology & Architecture

### The Model: ConvNeXt V2
Instead of traditional ResNets or pure Vision Transformers (ViTs), this project utilizes **ConvNeXt V2** (accessed via `timm`).

* **Why ConvNeXt V2?** It modernizes standard ConvNets with design choices inspired by Transformers (e.g., larger kernel sizes, layer normalization strategies).
* **Key Feature:** It incorporates **Global Response Normalization (GRN)** layers, which enhance inter-channel competition. This is particularly effective for feature co-adaptation, preventing the model from collapsing into feature redundancy.
* **Transfer Learning:** We utilize pre-trained weights (typically on ImageNet) and fine-tune the head to our specific snake species classes, allowing for faster convergence and better generalization on smaller datasets.

### Hyperparameter Optimization
To avoid manual tuning bias, **Optuna** is integrated into the workflow. We perform a Bayesian search to optimize key hyperparameters:
* **Learning Rate & Scheduler:** Finding the optimal convergence trajectory.
* **Batch Size:** Balancing memory usage and gradient stability.
* **Augmentation Intensity:** Tuning the strength of geometric and color distortions to prevent overfitting.
