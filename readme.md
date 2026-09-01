# Deep Neural Networks Architectures from the Perspective of Manifold Learning

A practical reproduction of the manifold-learning and topological-analysis methodology presented in the research paper:

> **German Magai, "Deep Neural Networks Architectures from the Perspective of Manifold Learning"**

## Project Overview

Deep neural networks transform input data through a sequence of internal representations. Instead of evaluating a neural network only through classification accuracy, this project studies how these internal representations change across network depth from a geometric and topological perspective.

The project implements the core analysis methodology of the research paper using:

- Deep Learning
- Manifold Learning
- Topological Data Analysis (TDA)
- Persistent Homology
- Vietoris-Rips filtration
- H₀ and H₁ homology
- Topological descriptors
- Persistent Homological Fractal Dimension (PHdim)

The implementation focuses on a **ResNet model trained on CIFAR-10** and analyzes the representations produced by its structural blocks.

The project is designed as a **practical methodological reproduction** of the paper rather than a complete replication of all architectures and experiments presented in the original research.

## Reference Research Paper

**German Magai**

*Deep Neural Networks Architectures from the Perspective of Manifold Learning*

The original research paper investigates the internal representations of deep neural networks from the perspective of manifold learning and topology.

It studies how the structure of neural-network representations changes across network depth and during training using topological analysis, persistent homology, and Persistent Homological Fractal Dimension (PHdim).

## Project Objective

The main objective of this project is to implement and demonstrate the core analysis pipeline described in the research paper on a computationally manageable ResNet + CIFAR-10 experiment.

The project aims to:

1. Train a CIFAR-10-compatible ResNet.
2. Extract internal representations from its structural blocks.
3. Apply Global Average Pooling to vectorize CNN feature maps.
4. Compute Vietoris-Rips persistent homology.
5. Analyze H₀ and H₁ topological structures.
6. Calculate persistence-based topological descriptors.
7. Estimate Persistent Homological Fractal Dimension (PHdim).
8. Analyze how these measurements change across network depth.
9. Compare the implemented methodology and observations with those reported in the original paper.

## Project Scope

The original research paper investigates multiple neural-network architecture families, including:

- ResNet
- SE-ResNet
- MobileNetV2
- VGG
- Vision Transformer
- ConvMixer
- BERT
- RoBERTa

In this project, we implemented **ResNet only**.

This is an intentional computational simplification. The purpose of this short project was to implement and test the core manifold-learning and topological-analysis methodology on one representative architecture rather than attempting to reproduce the complete experimental study.

## Why ResNet Only?

The complete methodology involves model training, extraction of high-dimensional internal representations, and persistent-homology calculations. Repeating these experiments across many architectures and training stages is computationally expensive.

Our implementation was performed in a **CPU-only environment**. Training and persistent-homology analysis therefore required considerable computation time.

We selected ResNet as a focused case study so that we could implement the complete analysis pipeline:

```text
Model Training
      ↓
Internal Representation Extraction
      ↓
Global Average Pooling
      ↓
Persistent Homology
      ↓
Topological Descriptors
      ↓
PHdim
      ↓
Layer-wise Analysis

## Dataset

### CIFAR-10

The experiment uses the **CIFAR-10** image-classification dataset.

The original CIFAR-10 dataset contains:

- 50,000 training images
- 10,000 test images
- 10 classes
- Image size: 32 × 32 RGB images

### Computational Subset

Because the experiment was performed in a CPU-only environment, we used a balanced subset of CIFAR-10:

| Dataset | Samples |
|---|---:|
| Training set | 5,000 |
| Test set | 1,000 |
| Training samples per class | 500 |
| Test samples per class | 100 |

The subset maintains equal representation across all ten classes.

This reduction was made purely for computational feasibility. It is not intended to replace the full dataset used in the original research.

The CIFAR-10 dataset itself is **not included in this repository**.

## Model Architecture

A custom CIFAR-10-compatible **ResNet** was implemented for this project.

The implementation follows the paper-supported characteristics used for the representation analysis:

- 10 structural blocks
- 128 feature channels
- Analysis of internal representations at the output of each structural block

The general flow of the network is:

```text
Input Image
     ↓
Initial Convolution
     ↓
Residual Block 1
     ↓
Residual Block 2
     ↓
     .
     .
     .
     ↓
Residual Block 10
     ↓
Global Average Pooling
     ↓
Fully Connected Layer
     ↓
10-Class Output

Each structural block produces an internal feature representation that is later extracted for the manifold and topological analysis.

Model Parameters:
Total parameters       : 2,959,242
Trainable parameters   : 2,959,242

The ResNet implementation is a practical reproduction. The original paper does not provide complete source code for the exact ResNet implementation used in its experiments, so the model used here is a CIFAR-10-compatible implementation designed to follow the relevant architectural characteristics described in the paper.
