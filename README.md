# Performance Benchmarking of Graph Neural Networks on CPU, GPU, and TPU

This repository contains the implementation and comparative analysis of training Graph Neural Networks (GNNs) on various computational platforms: **CPU**, **GPU**, and **TPU**. The goal of the project is to evaluate and understand the training efficiency, accuracy, and resource utilization across these hardware platforms.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Graph Neural Networks (GNNs) are widely used for solving graph-structured problems, but their training can be computationally expensive. In this project, we implement and train GNN models on different hardware platforms and analyze their performance in terms of:
- Training time
- Model accuracy
- Resource utilization

The project uses PyTorch and PyTorch Geometric libraries for implementing the GNN models, focusing on a **Signed Graph Convolutional Network (SignedGCN)** architecture.

---

## Features

- Implementation of SignedGCN for graph-based tasks.
- Comparative analysis of training on:
  - CPU (Colab Standard)
  - GPU (NVIDIA T4, Colab)
  - TPU (v2-8, Colab)
- Automatic logging of training time and accuracy for each platform.
- Preprocessing and spectral feature generation for graph datasets.

---

## Installation

### Prerequisites
Ensure you have Python 3.8 or higher installed along with the following dependencies:
- `torch`
- `torch-geometric`
- `torch-xla` (for TPU support)
- `numpy`

### Installing Dependencies
Run the following commands in your environment:
```bash
# Clone the repository
git clone https://github.com/your_username/gnn-performance-benchmark.git
cd gnn-performance-benchmark

# Install dependencies
pip install torch torch-geometric torch-xla numpy
Usage
Running on CPU
bash
Copy code
python run_cpu.py
Running on GPU
Ensure GPU runtime is enabled in your environment:

bash
Copy code
python run_gpu.py
Running on TPU
Enable TPU runtime in Google Colab or similar platforms, then run:

bash
Copy code
python run_tpu.py
Example Output
Training time for CPU: 28.37 seconds
Training time for GPU: 8.30 seconds
Training time for TPU: 957.76 seconds
Results
Training Time Comparison
Hardware Platform	Training Time (seconds)
CPU	28.37
GPU	8.30
TPU	957.76
Accuracy Observations
All platforms achieved comparable accuracy in testing, demonstrating hardware differences primarily impact training speed.

Contributing
We welcome contributions! Please follow these steps:

Fork this repository.
Create a new branch (git checkout -b feature/your-feature-name).
Commit your changes (git commit -am 'Add some feature').
Push to the branch (git push origin feature/your-feature-name).
Open a Pull Request.
License
This project is licensed under the MIT License.

Acknowledgments
PyTorch and PyTorch Geometric for providing GNN libraries.
Google Colab for computational resources.
The community for support and feedback.
Contact
For any questions or suggestions, feel free to reach out:

Email: your_email@example.com
GitHub Issues: Submit an Issue
markdown
Copy code

### Notes:
- Replace `your_username` and `your_email@example.com` with your actual GitHub username and email.
- Include a valid license file (`LICENSE`) if you’re using the MIT License.
