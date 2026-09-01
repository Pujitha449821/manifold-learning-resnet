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
```

Each structural block produces an internal feature representation that is later extracted for the manifold and topological analysis.

### Model Parameters

```text
Total parameters       : 2,959,242
Trainable parameters   : 2,959,242
```

The ResNet implementation is a practical reproduction.

## Training

The ResNet model was trained on the balanced CIFAR-10 subset using a CPU-based environment.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Learning Rate | 0.001 |
| Batch Size | 128 |
| Device | CPU |
| Training Epochs | 1 |

### Training Results

The one-epoch experiment achieved:

| Metric | Result |
|---|---:|
| Training Accuracy | 25.36% |
| Test Accuracy | 23.50% |

The trained model checkpoint is:

```text
models/resnet_cifar10_epoch1.pth
```

### Important Reproduction Note

The original research paper trained its models until they reached at least approximately 95% training accuracy.

In this project, training was intentionally limited to one epoch because of the available CPU-only computational resources.

Therefore, the model obtained in this experiment is not considered a fully trained reproduction of the models in the original paper. Instead, it is used to implement and validate the subsequent representation-extraction and topological-analysis pipeline.

The ResNet implementation is a practical reproduction. The original paper does not provide complete source code for the exact ResNet implementation used in its experiments, so the model used here is a CIFAR-10-compatible implementation designed to follow the relevant architectural characteristics described in the paper.

## Internal Representation Extraction

One of the main objectives of this project is to study how the internal representation of data changes as it passes through the ResNet.

Instead of analyzing only the final classification output, representations were extracted from **all 10 structural blocks**.

Each block initially produces a feature map with the shape:

```text
[batch, 128, 32, 32]
```

Following the representation-analysis methodology described in the research paper, **Global Average Pooling (GAP)** was applied to vectorize the CNN feature maps:

```text
[batch, 128, 32, 32]
          ↓
Global Average Pooling
          ↓
[batch, 128]
```

The resulting representation from each block therefore contains 128 features per sample.

For the 1,000 test samples used in the analysis, the representations have the following structure:

```text
Block 1  → [1000, 128]
Block 2  → [1000, 128]
Block 3  → [1000, 128]
Block 4  → [1000, 128]
Block 5  → [1000, 128]
Block 6  → [1000, 128]
Block 7  → [1000, 128]
Block 8  → [1000, 128]
Block 9  → [1000, 128]
Block 10 → [1000, 128]
```

Thus, each structural block can be viewed as a **128-dimensional point cloud containing 1,000 observations**.

These internal representations form the input for the subsequent persistent-homology and topological analysis.

## Representation Verification

Before performing the final persistent-homology analysis, the extracted internal representations were verified to ensure that they were numerically valid and had the expected structure.

The verification checked the following:

- No NaN values
- No infinite values
- 1,000 samples per structural block
- 128 features per sample
- All 10 CIFAR-10 classes represented
- 100 test samples per class
- Consecutive block representations were numerically different

The verification confirmed that the extracted representations had the expected shape:

```text
[1000, 128]
```

for each of the 10 structural blocks.

The feature statistics also changed progressively across the network, indicating that the ResNet transforms the input representations as data passes through successive structural blocks.

These verified representations were then used as the input point clouds for persistent-homology analysis.

## Persistent Homology

Persistent homology was used to study the topological structure of the internal ResNet representations.

The implementation uses the **Ripser** library and constructs Vietoris-Rips filtrations from the extracted representation point clouds.

The analysis focuses on two homology dimensions:

- **H₀** — connected components
- **H₁** — one-dimensional cycles or loop-like structures

For each structural block, the 128-dimensional representations were treated as a point cloud. Persistent homology tracks how topological features appear and disappear as the distance scale changes.

### Computational Sampling

Persistent-homology computation becomes increasingly expensive as the number of points and dimensionality increase.

Therefore, after testing the implementation on smaller samples, a common sample of **300 points** was used across all 10 structural blocks for the final layer-wise topology analysis.

Using the same sample size across blocks allows the topological measurements to be compared consistently across network depth.

### Persistent Homology Workflow

```text
128-Dimensional Representations
            ↓
       Point Cloud
            ↓
   Vietoris-Rips Filtration
            ↓
   Persistent Homology
            ↓
       H₀ and H₁
            ↓
   Persistence Diagrams
```

The resulting persistence information was then used to calculate the topological descriptors described in the next stage of the analysis.

## Topological Descriptors

The persistence diagrams obtained from the persistent-homology analysis were converted into numerical descriptors based on the lifespans of the persistent features.

For each persistence pair, the lifespan is defined as:

```text
lifespan = death − birth
```

For this project, the lifespan-sum descriptor was calculated for both H₀ and H₁ using:

```text
α = 1
```

The descriptors summarize the persistence of topological features within each network block.

The analysis therefore produces:

- H₀ topological descriptor for each of the 10 structural blocks
- H₁ topological descriptor for each of the 10 structural blocks

These numerical descriptors allow the topological properties of the internal representations to be compared across network depth.

The resulting values are stored in the `results/tables/` directory and are used for the layer-wise analysis and visualization.

### Descriptor Calculation

```text
Persistence Diagram
        ↓
Birth–Death Pairs
        ↓
Lifespan = Death − Birth
        ↓
Sum of Lifespans
        ↓
H₀ / H₁ Topological Descriptor
        ↓
Comparison Across Network Depth
```

These descriptors provide a quantitative way to study how the topology of the ResNet representations changes from the first structural block to the final structural block.
