# Spatially-Gated Adaptive Multi-Expert Learning and Uncertainty-Guided GCN for Multi-Focus Image Fusion

> **Abstract / Introduction**
> This repository contains the official PyTorch implementation of the paper: *"Spatially-Gated Adaptive Multi-Expert Learning and Uncertainty-Guided GCN for Multi-Focus Image Fusion"*. 
> 
> We propose a novel heterogeneous collaborative decoding framework that unifies local detail modeling and global structural reasoning. By designing an Adaptive Mixture of Feature Experts and an uncertainty-guided graph construction mechanism, our network effectively overcomes the long-standing challenges of high-frequency texture attenuation and topological inconsistency in multi-focus image fusion.

---

## Key Features

*   **Adaptive Mixture of Feature Experts:** Employs a pixel-level sparse routing mechanism to dynamically dispatch spatial, frequency (via FFT), and edge experts, maximizing the fidelity of sharp image textures and suppressing artifacts in flat regions.
*   **Uncertainty-Guided Graph Construction:** Utilizes Laplacian uncertainty-driven adaptive sampling to anchor structural mutation regions, breaking the $\mathcal{O}(N^2)$ computational bottleneck of global graph construction for high-resolution images.
*   **GCN-GAT Cascaded Encoder:** Performs topological "structure purification" and "noise suppression," seamlessly transmitting global semantic context while denoising features.

---

## Prerequisites

*   Python $\ge$ 3.8
*   PyTorch $\ge$ 1.10.0
*   Torchvision
*   NumPy, OpenCV, SciPy

```bash
# Clone this repository
git clone https://github.com/DengMinyu/FDMM-SCFusion.git
cd Fusion-Net

# Install dependencies
pip install -r requirements.txt