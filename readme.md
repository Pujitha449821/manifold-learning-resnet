#Project Overview

This project presents a practical reproduction of the manifold-learning and topological-analysis methodology proposed in the research paper:

"Deep Neural Networks Architectures from the Perspective of Manifold Learning"
German Magai

The project studies how the internal representation of data changes as it passes through the structural blocks of a Deep Neural Network (DNN).

Instead of evaluating a neural network only through classification accuracy, the project analyzes the geometry and topology of internal representations using:

Manifold Learning
Topological Data Analysis (TDA)
Persistent Homology
Topological Descriptors
Persistent Homological Fractal Dimension (PHdim)

The implementation focuses on the ResNet-based CIFAR-10 experiment from the paper and adapts the methodology to a CPU-based computational environment.

The research paper describes the evolution of embedding manifolds across the depth of neural networks and uses topological descriptors and PHdim to characterize these changes.

#Objective

The main objectives of this project are:

Analyze how data representations change across network depth.
Extract internal representations from the structural blocks of a neural network.
Study changes in the topology of these representations.
Calculate persistent-homology-based topological descriptors.
Estimate geometric complexity using PHdim.
Reproduce the core computational methodology of the research paper under practical computational constraints.

#Methodology

The implemented pipeline is:

                 CIFAR-10
                     |
                     v
             10-Block ResNet
                     |
                     v
       Internal Block Representations
                     |
                     v
          Global Average Pooling
                     |
                     v
              128-D Point Clouds
                     |
                     v
          Vietoris-Rips Filtration
                     |
                     v
        Persistent Homology (H0, H1)
                     |
                     v
          Topological Descriptors
                     |
                     v
                  PHdim
The paper analyzes the representation at the output of each structural block and uses Global Average Pooling to vectorize CNN feature maps before topological analysis.

#Dataset
CIFAR-10

CIFAR-10 is an image-classification dataset containing:

10 classes
50,000 training images
10,000 test images
Image size: 32 × 32
3 RGB channels
5,000 training images per class
1,000 test images per class

The dataset contains the following classes:
0  Airplane
1  Automobile
2  Bird
3  Cat
4  Deer
5  Dog
6  Frog
7  Horse
8  Ship
9  Truck

The research paper uses CIFAR-10 among its image-classification datasets.

Practical Dataset Used

Because the experiment was performed in a CPU-only environment, a balanced subset was used for the initial computational reproduction:

Training samples : 5,000
Test samples     : 1,000
Training/class   : 500
Test/class       : 100
The coding implementation verified that all ten classes were represented equally in the 1,000-image test representation.

#Model Architecture

A custom CIFAR-10-compatible ResNet was implemented.

The research paper specifies that its CNN experiments use:

10 structural blocks
128 channels
Global Average Pooling for CNN representations

This configuration was used to reduce the influence of the ambient space when comparing architectures.

Implemented Architecture
Input Image | v Initial Convolution | Batch Normalization + ReLU | v Residual Block 1 | Residual Block 2 | . . . | Residual Block 10 | v Global Average Pooling | v Fully Connected Layer | v 10-Class Output

Each residual block contains convolutional operations, batch normalization, ReLU activation and an identity/skip connection.

The paper does not provide sufficient implementation details to establish that this exact ResNet is identical to the author's implementation. Therefore, this project treats the model as a paper-consistent practical reproduction.

Model Size
Total parameters      : 2,959,242
Trainable parameters  : 2,959,242

This parameter count belongs to the implementation in this project and is not claimed to be the parameter count of the original paper's model.

#Training

The model was initially trained using:

Parameter	Value
Optimizer	Adam
Learning Rate	0.001
Loss Function	Cross-Entropy Loss
Batch Size	128
Device	CPU
Training Samples	5,000
Test Samples	1,000
Epochs	1
The paper states that its models were trained using Adam with an adaptive learning rate until reaching at least 95% training accuracy.

Initial Training Results
Metric	Result
Training Loss	2.1136
Training Accuracy	25.36%
Test Loss	2.1034
Test Accuracy	23.50%
Training Time	21.53 minutes

The one-epoch experiment is therefore considered an initial computational/development run, rather than a complete numerical reproduction of the paper's final training condition.
#Internal Representation Extraction

The objective of representation extraction is to observe how the data representation changes after each structural block.

The output of all 10 residual blocks was extracted.

Before vectorization, the feature maps have the form:

[batch, 128, 32, 3]
Global Average Pooling was then applied:

[batch, 128, 32, 32]
          |
          v
Global Average Pooling
          |
          v
[batch, 128]

After processing 1,000 test samples, each block produced:

Block 1  -> [1000, 128]
Block 2  -> [1000, 128]
Block 3  -> [1000, 128]
Block 4  -> [1000, 128]
Block 5  -> [1000, 128]
Block 6  -> [1000, 128]
Block 7  -> [1000, 128]
Block 8  -> [1000, 128]
Block 9  -> [1000, 128]
Block 10 -> [1000, 128]
Thus, each block produces a 128-dimensional point cloud containing 1,000 observations.

Memory Optimization

During the initial implementation, storing complete feature maps from all ten blocks caused excessive memory usage.

Therefore, Global Average Pooling was performed immediately after each block instead of storing the complete feature maps.

This reduced memory consumption while preserving the required 128-dimensional representation and following the CNN vectorization procedure described in the paper.
#Representation Verification

The extracted representations were verified before applying persistent homology.

The verification confirmed:

No NaN values.
No infinite values.
1,000 samples per block.
128 features per sample.
All 10 CIFAR-10 classes were represented.
Each class contained 100 test samples.
Consecutive block representations were numerically different.

The feature statistics changed progressively through network depth, demonstrating that the network transforms the representations as data passes through its structural blocks.

#Persistent Homology

Persistent Homology was used to study the topology of the internal representations.

The implementation uses:

Vietoris-Rips filtration
H₀ — connected components
H₁ — one-dimensional cycle structures
Ripser

The paper also restricts its analysis to homology dimensions 0 and 1 because higher-dimensional persistent-homology calculations are computationally expensive.
100-Point Verification

A 100-point sample from Block 1 was first used to verify the implementation.

Input shape             : (100, 128)

H0 persistence pairs    : 100
Finite H0 pairs         : 99

H1 persistence pairs    : 27
Finite H1 pairs         : 27

This confirmed that Ripser could successfully calculate the required persistence diagrams.

300-Point Scaling Test

A larger 300-point sample was subsequently tested:

Input shape             : (300, 128)

H0 persistence pairs    : 300
Finite H0 pairs         : 299

H1 persistence pairs    : 143
Finite H1 pairs         : 143
Runtime                 : approximately 0.04 seconds

This established that a 300-point sample was computationally manageable in the CPU environment.


#opological Descriptors

Persistence diagrams were converted into numerical topological descriptors using the lifespan-sum formulation with:

α = 1

For a persistence feature:

lifespan = death - birth

The descriptor is calculated by summing the lifespans of finite persistence features.

The calculation was performed separately for:

H0 → Connected-component structure
H1 → One-dimensional cycle structure

The layer-wise experiment produced:
10 H0 descriptors
10 H1 descriptors

corresponding to the ten ResNet structural blocks.

A decrease in the H₀ descriptor can indicate increasing compactness of the representations, while a decrease in H₁ can indicate a reduction in noticeable cycle-like structures.

#Persistent Homological Fractal Dimension (PHdim)

PHdim was used to estimate the geometric complexity of the representation based on the scaling behavior of persistent homology.

For the pilot experiment:

Homology dimension : H0
α                   : 1
Block               : 1
Sample sizes        : {50, 100, 150, 200, 250, 300}

The relationship

log10(E) = β log10(n) + c

was fitted using linear regression.

The resulting values were:

β  = 0.8062056
R² = 0.9979870

Using:

PHdim = 1 / (1 - β)
the resulting estimate was:

PHdim ≈ 5.16

The coding experiment obtained a more precise value of approximately 5.1601.

This should be interpreted as a finite-sample estimate for the implemented Block 1 experiment, not as an exact reproduction of the paper's reported PHdim values.

#Generated Outputs

The implementation generated the following saved PyTorch files:

resnet_epoch1_vectorized_representations.pt
resnet_epoch1_labels.pt
resnet_epoch1_topological_descriptors.pt

These files allow subsequent analysis without repeating the computationally expensive ResNet forward pass.

Recommended Repository Location

For the final GitHub repository, the files should be organized as:

results/
└── representations/
    ├── resnet_epoch1_vectorized_representations.pt
    └── resnet_epoch1_labels.pt

results/
└── persistence/
    └── resnet_epoch1_topological_descriptors.pt


#Repository Structure






















#Code Organization

The code is organized according to the main stages of the experiment:

code/training/

Contains the implementation used to:

Load the CIFAR-10 training subset.
Construct the ResNet.
Train the network.
Evaluate training and test performance.
Save the trained model if applicable.
code/representation_extraction/

Contains the implementation used to:

Load the trained ResNet.
Process the CIFAR-10 test subset.
Extract outputs from all ten residual blocks.
Apply Global Average Pooling.
Produce the 1,000 × 128 representations.
code/persistent_homology/
Contains the implementation used to:

Load vectorized representations.
Sample observations.
Calculate Vietoris-Rips persistent homology.
Calculate H₀ and H₁ persistence diagrams.
Compute topological descriptors.
code/phdim/

Contains the implementation used to:

Evaluate different sample sizes.
Calculate H₀ persistence.
Obtain lifespan sums.
Perform log-log regression.
Estimate PHdim.

#Reproduction Workflow

The complete experiment can be understood as the following sequence:

1. Environment Setup
        ↓
2. CIFAR-10 Loading
        ↓
3. ResNet Construction
        ↓
4. Model Training
        ↓
5. Internal Representation Extraction
        ↓
6. Representation Verification
        ↓
7. Persistent Homology
        ↓
8. Topological Descriptors
        ↓
9. PHdim
        ↓
10. Results and Interpretation
The coding report follows these same stages from environment setup through PHdim calculation.

Environment

The experiment was developed using:

Python             3.10.20
Anaconda env       manifold_learning
Jupyter Notebook
PyTorch            2.13.0+cpu
Torchvision        0.28.0+cpu
NumPy              2.2.6
Pandas
Matplotlib
Scikit-learn
SciPy
Ripser             0.6.15
Persim

The implementation was executed using CPU-based PyTorch because CUDA was unavailable in the development environment.
Installation

Create the environment and install the dependencies listed in:

requirements.txt

setup is:

conda create -n manifold_learning python=3.10.20
conda activate manifold_learning
pip install -r requirements.txt

Then launch Jupyter Notebook if required:

jupyter notebook

Computational Considerations

The original paper reports experiments using an NVIDIA GeForce GTX 3080Ti, Intel Core i7-10700K and 16 GB RAM.

This project was executed in a CPU-only environment.

Because of the computational limitations:

A reduced CIFAR-10 subset was used for the initial training experiment.
The model was trained for one epoch.
300 points were used for the layer-wise persistent-homology experiment.
A pilot PHdim experiment was performed on Block 1.

The one-epoch training experiment required approximately 21.53 minutes even with the reduced dataset.

These changes are computational adaptations and do not represent parameters claimed to be identical to those of the original paper.
Relation to the Research Paper

The research paper investigates internal representations across several neural-network architecture families, including:

ResNet
SE-ResNet
MobileNetV2
VGG
Vision Transformer
ConvMixer
BERT
RoBERTa

For its CNN experiments, the paper uses ten structural blocks and 128 channels and applies Global Average Pooling to vectorize feature maps.

This project focuses specifically on the ResNet + CIFAR-10 setting.

The main methodology is preserved:
Internal representations
        ↓
Global Average Pooling
        ↓
Persistent Homology
        ↓
Topological Descriptors
        ↓
PHdim

However, the implementation is a paper-consistent practical reproduction, not an exact numerical replication.

#Limitations

The current implementation differs from the original paper in several ways:

A reduced CIFAR-10 subset was used for the initial training experiment.
The initial model was trained for only one epoch.
CPU computation was used instead of the reported GPU hardware.
A single common 300-point sample was used for the layer-wise persistent-homology analysis.
The exact original ResNet implementation is not available from the paper.
The original paper uses repeated class-wise sampling and averaging, whereas the current implementation uses a practical single-sample strategy.
The PHdim calculation is currently a Block 1 finite-sample pilot experiment.

The paper's experimental protocol uses randomly selected batches of 300 examples from the training data of each class and averages measurements across batches and classes.

Therefore, the numerical values produced by this project should be interpreted as results of a methodological and computational reproduction, rather than direct reproduction of the paper's reported numerical results.

#Key Results

The current implementation successfully established the complete computational pipeline.

ResNet
Structural blocks       : 10
Feature channels        : 128
Parameters              : 2,959,242
Representation Extraction
Samples per block       : 1,000
Dimensions per sample   : 128

Representation shape:
[1000, 128]
Persistent Homology

For the 300-point Block 1 experiment:

H0 pairs                 : 300
Finite H0 pairs          : 299
H1 pairs                 : 143
Finite H1 pairs          : 143
PHdim
β                        : 0.8062056
R²                       : 0.9979870
PHdim                    : approximately 5.16
These results demonstrate that the complete representation → persistent homology → descriptor → PHdim pipeline was successfully implemented.

Project Report

The detailed implementation process, verification results and computational observations are documented in the project report located in:

report/

The report covers:

Environment Setup
       ↓
CIFAR-10 Preparation
       ↓
ResNet Construction
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

#Conclusion

This project implements the core computational methodology proposed in the research paper for studying deep neural-network representations from a manifold-learning perspective.

A custom 10-block, 128-channel ResNet was implemented for CIFAR-10. Internal representations were extracted from all structural blocks and converted into 128-dimensional point clouds using Global Average Pooling.

Persistent homology was then applied to analyze H₀ and H₁ structures, followed by calculation of topological descriptors and Persistent Homological Fractal Dimension.

The Block 1 pilot experiment produced:

PHdim ≈ 5.16

Although computational limitations prevented an exact reproduction of the paper's full training and sampling protocol, the project successfully demonstrates the central methodology of studying how the geometry and topology of neural-network representations evolve with depth.

Reference

Magai, German.

Deep Neural Networks Architectures from the Perspective of Manifold Learning.

team members 
Pujitha V
Bhavana D
Soumithri Ch


 
