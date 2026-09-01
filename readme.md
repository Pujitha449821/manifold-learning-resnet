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
