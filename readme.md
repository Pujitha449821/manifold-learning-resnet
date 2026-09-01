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

## Persistent Homological Fractal Dimension (PHdim)

Persistent Homological Fractal Dimension (PHdim) was implemented to quantify the scaling behavior of the persistent-homology descriptors as the number of sampled points increases.

For the PHdim analysis in this project, the following configuration was used:

- Homology dimension: H₀
- Power parameter: α = 1
- Sample sizes: 50, 100, 150, 200, 250, 300

The H₀ persistence descriptor was calculated for each sample size. The scaling relationship was then analyzed using logarithmic transformation and linear regression.

The fitted relationship is:

```text
log10(E₁⁰) = β log10(n) + c
```

where:

- `n` is the number of sampled points
- `E₁⁰` is the H₀ persistence-based descriptor
- `β` is the scaling exponent
- `c` is the regression intercept

PHdim was calculated using the formulation:

```text
PHdim = α / (1 − β)
```

For the Block 1 pilot experiment, the implementation produced:

```text
β  = 0.8062056
R² = 0.9979870
PHdim ≈ 5.16
```

The PHdim calculation was first verified on Block 1 and was then applied to the representations from all 10 structural blocks.

The resulting PHdim values are used to study how the estimated geometric complexity of the representations changes across network depth.

These values should be interpreted as estimates obtained from the computationally constrained experiment and are not claimed to be exact reproductions of the paper's reported values.

## Layer-wise Analysis

The internal representations were analyzed across all 10 structural blocks of the ResNet.

To compare the representations according to network depth, the structural blocks were represented using relative depth:

```text
Block 1  → 0.1
Block 2  → 0.2
Block 3  → 0.3
Block 4  → 0.4
Block 5  → 0.5
Block 6  → 0.6
Block 7  → 0.7
Block 8  → 0.8
Block 9  → 0.9
Block 10 → 1.0
```

For every structural block, the following quantities were analyzed:

- H₀ topological descriptor
- H₁ topological descriptor
- PHdim

The same 300-point sampling strategy was used across the blocks so that the resulting topological measurements could be compared consistently.

The layer-wise analysis produces both numerical tables and visualizations. The numerical results are stored in:

```text
results/tables/
```

and the generated plots are stored in:

```text
results/figures/
```

The resulting measurements allow us to examine how the topological and geometric properties of the learned representations evolve from shallow to deeper parts of the network.

## Results

The completed experiment produced layer-wise measurements of the internal ResNet representations using persistent homology and PHdim.

The main quantities analyzed across the 10 structural blocks are:

- H₀ topological descriptor
- H₁ topological descriptor
- PHdim

The experiment also generated visualizations showing how these measurements change with relative network depth.

### Generated Figures

The main figures are available in:

```text
results/figures/
```

They include:

- `resnet_epoch1_H0_vs_depth.png`
- `resnet_epoch1_H1_vs_depth.png`
- `resnet_epoch1_PHdim_vs_depth.png`
- `resnet_epoch1_normalized_topology.png`

### Numerical Results

The corresponding numerical outputs are available in:

```text
results/tables/
```

These include:

- `resnet_epoch1_accuracy_topology_summary.csv`
- `resnet_epoch1_final_graph_data.csv`
- `resnet_epoch1_layerwise_report_table.csv`
- `resnet_epoch1_layerwise_topology.csv`

Additional analysis output is available in:

```text
results/analysis/
```

including:

- `resnet_epoch1_correlation_summary.csv`

### Interpretation

The results demonstrate that the internal representations of the ResNet exhibit measurable changes in their topological properties across network depth.

The H₀ and H₁ descriptors provide quantitative measurements of persistent topological structures, while PHdim provides an additional measure based on the scaling behavior of the persistence descriptor.

Because this experiment was limited to one model and one training epoch, the observed layer-wise behavior should be interpreted as the result of this specific computational experiment rather than as a general conclusion about fully trained neural networks.

The results are therefore presented primarily as a demonstration of the implemented manifold-learning and topological-analysis methodology.

## Accuracy and Topology Analysis

The project also examined the available model-performance and topological measurements together.

The trained ResNet produced the following performance after one epoch:

| Metric | Result |
|---|---:|
| Training Accuracy | 25.36% |
| Test Accuracy | 23.50% |

The topology analysis was performed on the internal representations extracted from the 10 structural blocks.

However, a statistically meaningful topology–accuracy correlation was **not established from this experiment**.

The main reason is that the experiment contains only:

```text
1 model
1 training epoch
1 trained checkpoint
```

A correlation across training requires multiple observations from different training states. The original paper analyzes topology and accuracy across multiple training stages and also considers multiple trained models for its broader generalization analysis.

Therefore, the correlation values reported in the original paper are **not presented as results of this project**.

The current experiment instead demonstrates how accuracy and topological measurements can be collected within the same analysis pipeline.

Reproducing the paper's topology–accuracy correlation would require additional training epochs and/or multiple independently trained models, which was outside the computational scope of this CPU-based short project.

## Comparison with the Research Paper

The implementation follows the main methodological ideas presented in the research paper, while reducing the experimental scope because of computational constraints.

| Aspect | Original Research Paper | This Project |
|---|---|---|
| Architecture scope | Multiple architectures | ResNet only |
| Dataset scope | Multiple datasets | CIFAR-10 |
| ResNet structural blocks | 10 | 10 |
| Feature channels | 128 | 128 |
| CNN vectorization | Global Average Pooling | Global Average Pooling |
| Persistent homology | Vietoris-Rips | Vietoris-Rips |
| Homology dimensions | H₀ and H₁ | H₀ and H₁ |
| Topological descriptors | Used | Implemented |
| PHdim | Used | Implemented |
| Training | Models trained to high training accuracy | 1 epoch |
| Dataset scale | Larger/full experimental settings | Balanced 5,000/1,000 subset |
| Computational environment | GPU-based experiments | CPU-only |
| Layer-wise analysis | Performed | Performed |
| Topology–accuracy analysis | Multiple training states/models | Not statistically established |

The project therefore reproduces the **core computational methodology** of the paper while reducing the experimental scope to make the implementation feasible in the available computational environment.

The differences in dataset size, training duration, architecture scope, and sampling strategy should be considered when comparing the results with those reported in the original research.

## What Was Followed from the Paper vs. Our Choices

The project distinguishes between the methodology derived from the research paper and the practical decisions made to make the experiment computationally feasible.

### Methodological Elements Followed from the Paper

The following aspects were based on the methodology described in the research paper:

- Analysis of internal neural-network representations
- Analysis across structural blocks
- 10 structural blocks for the ResNet analysis
- 128 feature channels
- Global Average Pooling for CNN feature-map vectorization
- Vietoris-Rips persistent homology
- H₀ and H₁ analysis
- Persistence-based topological descriptors
- Persistent Homological Fractal Dimension (PHdim)
- Layer-wise analysis of representations

### Practical Choices Made in This Project

The following choices were made specifically for this computational reproduction:

- ResNet-only architecture scope
- CIFAR-10-only dataset scope
- 5,000 training samples
- 1,000 test samples
- 500 training samples per class
- 100 test samples per class
- One training epoch
- Adam optimizer
- Learning rate of 0.001
- Batch size of 128
- CPU-based training
- 300-point sample for the final layer-wise persistent-homology analysis

These choices were made because the project was developed in a CPU-only environment and the complete experimental scope of the research paper would require substantially greater computational resources.

Therefore, the project should be viewed as a focused implementation of the paper's methodology rather than a full replication of every experiment.

## Limitations

The current implementation has several limitations that should be considered when interpreting the results.

### 1. Single Architecture

Only ResNet was implemented, whereas the original research paper investigates multiple neural-network architectures.

### 2. Reduced Dataset

A balanced subset of CIFAR-10 was used instead of the full dataset in order to keep training computationally manageable.

### 3. One Training Epoch

The model was trained for only one epoch because of the available CPU-only computational resources.

The original paper trains its models to substantially higher training accuracy.

### 4. Limited Persistent-Homology Sample Size

A common 300-point sample was used for the final layer-wise persistent-homology analysis.

This was necessary because persistent-homology computation becomes increasingly expensive as the number of points and representation dimensionality increase.

### 5. Single Training Checkpoint

Only one trained checkpoint was available.

Consequently, the project cannot establish the paper's multi-epoch topology–accuracy relationship.

### 6. No Paper-Style Correlation Result

The topology–accuracy correlations reported in the original research were not reproduced in this project because multiple training states and/or multiple trained models would be required.

### 7. Practical ResNet Implementation

The exact source implementation used in the original paper was not available in sufficient detail to establish an identical implementation.

Therefore, a CIFAR-10-compatible ResNet was implemented following the relevant architectural characteristics described in the paper.

### Overall Limitation

These constraints mean that this project should be considered a **practical methodological reproduction and implementation study**, rather than a complete numerical replication of the original research paper.

## Repository Structure

The repository is organized to separate the model checkpoint, experimental notebooks, and generated results.

```text
manifold-learning-resnet/
│
├── models/
│   └── resnet_cifar10_epoch1.pth
│
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_cifar10_preprocessing.ipynb
│   ├── 03_resnet_setup.ipynb
│   ├── 04_training.ipynb
│   ├── 05_representation_extraction.ipynb
│   ├── 06_representation_verification.ipynb
│   ├── 07_persistent_homology.ipynb
│   ├── 08_topological_descriptors.ipynb
│   ├── 09_phdim.ipynb
│   ├── 10_layerwise_analysis.ipynb
│   ├── 11_accuracy_topology_analysis.ipynb
│   ├── 12_graphs.ipynb
│   └── 13_correlation_analysis.ipynb
│
├── results/
│   ├── analysis/
│   │   └── resnet_epoch1_correlation_summary.csv
│   │
│   ├── figures/
│   │   ├── resnet_epoch1_H0_vs_depth.png
│   │   ├── resnet_epoch1_H1_vs_depth.png
│   │   ├── resnet_epoch1_PHdim_vs_depth.png
│   │   └── resnet_epoch1_normalized_topology.png
│   │
│   └── tables/
│       ├── resnet_epoch1_accuracy_topology_summary.csv
│       ├── resnet_epoch1_final_graph_data.csv
│       ├── resnet_epoch1_layerwise_report_table.csv
│       └── resnet_epoch1_layerwise_topology.csv
│
├── .gitignore
├── readme.md
└── requirements.txt
```

### Folder Description

| Folder/File | Purpose |
|---|---|
| `models/` | Contains the trained ResNet checkpoint |
| `notebooks/` | Contains the complete experimental workflow |
| `results/analysis/` | Contains additional analysis outputs |
| `results/figures/` | Contains the generated topology plots |
| `results/tables/` | Contains numerical result tables |
| `.gitignore` | Excludes datasets, intermediate files, and temporary files |
| `requirements.txt` | Lists the Python dependencies required for the project |
| `readme.md` | Project documentation |

The CIFAR-10 dataset and intermediate `.pt` tensor files are intentionally excluded from the repository because they are not required to understand the final reported results and would increase repository size.

## Notebook Execution Order

The notebooks are organized as a sequential experimental pipeline. They should generally be executed in the following order:

```text
01 → Environment Setup
02 → CIFAR-10 Preprocessing
03 → ResNet Setup
04 → Training
05 → Representation Extraction
06 → Representation Verification
07 → Persistent Homology
08 → Topological Descriptors
09 → PHdim
10 → Layer-wise Analysis
11 → Accuracy / Topology Analysis
12 → Graph Generation
13 → Correlation Analysis
```

### Notebook Descriptions

| Notebook | Purpose |
|---|---|
| `01_environment_setup.ipynb` | Set up and verify the Python environment |
| `02_cifar10_preprocessing.ipynb` | Load and prepare the CIFAR-10 subset |
| `03_resnet_setup.ipynb` | Define and verify the ResNet architecture |
| `04_training.ipynb` | Train and evaluate the ResNet |
| `05_representation_extraction.ipynb` | Extract representations from the 10 structural blocks |
| `06_representation_verification.ipynb` | Verify representation shapes and numerical validity |
| `07_persistent_homology.ipynb` | Compute H₀ and H₁ persistent homology |
| `08_topological_descriptors.ipynb` | Calculate persistence-based topological descriptors |
| `09_phdim.ipynb` | Calculate Persistent Homological Fractal Dimension |
| `10_layerwise_analysis.ipynb` | Analyze topology across network depth |
| `11_accuracy_topology_analysis.ipynb` | Examine available accuracy and topology measurements |
| `12_graphs.ipynb` | Generate final figures and result tables |
| `13_correlation_analysis.ipynb` | Analyze correlation-related outputs and document the limitations of the available data |

The notebooks together form the complete experimental workflow used in this project.

## Environment & Requirements

The project was developed and tested in a CPU-only environment.

### Main Software Requirements

- Python 3.10
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy
- Ripser
- Persim
- Jupyter Notebook

The dependency information used for the project is provided in:

```text
requirements.txt
```

The notebooks are designed to be executed in a Python environment with the required dependencies installed.

### Hardware Consideration

The experiments were performed using CPU computation. No CUDA/GPU acceleration was used for the reported experiment.

Because persistent-homology calculations can be computationally expensive for high-dimensional point clouds, the project uses reduced dataset and sampling sizes as described in the earlier sections.

## Installation and How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Pujitha449821/manifold-learning-resnet.git
cd manifold-learning-resnet
```

### 2. Create the Python Environment

Using Conda:

```bash
conda create -n manifold_learning python=3.10
conda activate manifold_learning
```

### 3. Install Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the `notebooks/` directory and execute the notebooks in the order described in the **Notebook Execution Order** section.

### 5. Dataset

The CIFAR-10 dataset is not included in this repository.

The dataset should be downloaded separately and prepared according to the preprocessing steps in:

```text
notebooks/02_cifar10_preprocessing.ipynb
```

### 6. Using the Existing Checkpoint

The trained one-epoch ResNet checkpoint is provided in:

```text
models/resnet_cifar10_epoch1.pth
```

This allows the later representation-extraction and topological-analysis stages to be examined without necessarily retraining the model.

### 7. Reproducing the Analysis

The complete workflow is:

```text
Environment Setup
       ↓
CIFAR-10 Preprocessing
       ↓
ResNet Setup
       ↓
Training
       ↓
Representation Extraction
       ↓
Representation Verification
       ↓
Persistent Homology
       ↓
Topological Descriptors
       ↓
PHdim
       ↓
Layer-wise Analysis
       ↓
Accuracy / Topology Analysis
       ↓
Graph Generation
       ↓
Correlation Analysis
```

The generated figures and numerical results are stored in the `results/` directory.

## Results and Output Files

The final experimental outputs are organized inside the `results/` directory.

### Figures

The `results/figures/` directory contains the main visualizations produced by the experiment:

- `resnet_epoch1_H0_vs_depth.png` — H₀ descriptor across relative network depth
- `resnet_epoch1_H1_vs_depth.png` — H₁ descriptor across relative network depth
- `resnet_epoch1_PHdim_vs_depth.png` — PHdim across relative network depth
- `resnet_epoch1_normalized_topology.png` — normalized combined topology analysis

### Tables

The `results/tables/` directory contains the numerical outputs used for reporting and visualization:

- `resnet_epoch1_accuracy_topology_summary.csv`
- `resnet_epoch1_final_graph_data.csv`
- `resnet_epoch1_layerwise_report_table.csv`
- `resnet_epoch1_layerwise_topology.csv`

### Analysis

The `results/analysis/` directory contains:

- `resnet_epoch1_correlation_summary.csv`

These files provide the numerical and visual outputs of the completed experiment and allow the reported results to be inspected without rerunning the complete computational pipeline.

The trained model checkpoint is separately available in:

```text
models/resnet_cifar10_epoch1.pth
```

Intermediate representation and persistent-homology tensor files are excluded from the repository through `.gitignore`.

## Computational Constraints

The complete experimental methodology described in the original research paper is computationally demanding, particularly when training multiple architectures and repeatedly calculating persistent homology on high-dimensional representations.

This project was developed in a **CPU-only environment**. Therefore, several practical reductions were necessary to make the experiment feasible.

The main computational adaptations were:

- Only ResNet was implemented.
- A balanced CIFAR-10 subset was used.
- The model was trained for only one epoch.
- A 300-point sample was used for the final layer-wise persistent-homology analysis.
- Only the required H₀ and H₁ homology dimensions were analyzed.

The one-epoch training experiment itself required approximately **21.53 minutes** in the CPU-based environment.

These computational constraints are important when interpreting the results. The experiment demonstrates the implementation of the methodology, but it does not reproduce the complete scale of the experiments performed in the original research paper.

A full reproduction would require substantially greater computational resources, longer training, multiple training checkpoints, and additional neural-network architectures.

## Conclusion

This project implemented the core manifold-learning and topological-analysis methodology described in the research paper for a ResNet-based CIFAR-10 experiment.

A custom 10-block ResNet with 128 feature channels was trained, and internal representations were extracted from all structural blocks. Global Average Pooling was used to convert the feature maps into 128-dimensional representations, which were then analyzed using Vietoris-Rips persistent homology.

The analysis included H₀ and H₁ persistence, persistence-based topological descriptors, and Persistent Homological Fractal Dimension (PHdim). These measurements were used to study how the internal representation changes across network depth.

Because the project was developed in a CPU-only environment, the experimental scope was intentionally reduced to one ResNet model, a balanced CIFAR-10 subset, one training epoch, and a 300-point sample for the final persistent-homology analysis.

Therefore, the results should be considered a **practical methodological reproduction** rather than a complete replication of the original paper's multi-architecture and multi-epoch experiments.

In particular, the available experiment does not provide enough training states or independently trained models to establish the topology–accuracy correlations reported in the original research.

Overall, the project demonstrates how persistent homology and related topological measurements can be used to quantitatively study the evolution of internal neural-network representations beyond conventional classification accuracy.

## Future Extensions

With additional computational resources, the project could be extended by:

- Training the ResNet for substantially more epochs.
- Collecting representations from multiple training checkpoints.
- Reproducing the topology–accuracy analysis across training.
- Training multiple ResNet-family configurations.
- Extending the analysis to the other architectures studied in the paper.
- Using larger datasets and larger persistent-homology samples.
- Performing repeated sampling and averaging for more robust estimates.
- Reproducing the broader multi-model experiments from the original research.

## Reference

**Magai, German.**

*Deep Neural Networks Architectures from the Perspective of Manifold Learning.*

This project is based on the methodology presented in the above research paper.

## Team Members

- **Pujitha V**
- **Bhavana D**
- **Soumithri Ch**
