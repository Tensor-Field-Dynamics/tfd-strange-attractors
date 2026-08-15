<div align="center">
  <img src="https://raw.githubusercontent.com/Tensor-Field-Dynamics/.github/main/assets/tfd_banner.png" alt="Tensor Field Dynamics Banner" width="100%">
  
  # High-Performance Strange Attractor Simulation Engine
  
  [![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Optimized-EE4C2C.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

  <p align="center">
    <b>GPU-accelerated numerical integration and rendering pipeline for deterministic-chaotic dynamical systems and topological data art.</b>
  </p>
</div>

---

## Table of Contents
- [Overview](#overview)
- [Mathematical Systems](#mathematical-systems)
  - [Clifford Attractor](#1-clifford-attractor-2d-discrete)
  - [Aizawa Attractor](#2-aizawa-attractor-3d-continuous)
  - [Peter de Jong Attractor](#3-peter-de-jong-attractor-2d-discrete)
- [Architecture & Pipeline](#architecture--pipeline)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Rendering & Post-Processing](#rendering--post-processing)
- [License](#license)

---

## Overview

This module provides a highly optimized computation and rendering engine to visualize chaotic phase spaces of continuous and discrete dynamical systems.

### Core Features
* **Fully Vectorized GPU Integration:** All trajectory computations are executed via native PyTorch tensors on the GPU to achieve maximum VRAM throughput without CPU bottlenecks.
* **Transient Chaos Filtering (Burn-in Phase):** Unstable transition states are eliminated by dynamically discarding initialization phases ($N_{\text{burn}}$), which isolates the pure strange attractor.
* **High-End Rendering Pipeline:** A histogram-based density accumulation method is utilized, featuring 4x Supersampling Anti-Aliasing (SSAA), volumetric shading, and filmic ACES tone mapping.

---

## Mathematical Systems

### 1. Clifford Attractor (2D Discrete)
The Clifford attractor is a two-dimensional, discrete iterative system defined by the following recurrence relations:

$$\begin{aligned} x_{n+1} &= \sin(a \cdot y_n) + c \cdot \cos(a \cdot x_n) \\ y_{n+1} &= \sin(b \cdot x_n) + d \cdot \cos(b \cdot y_n) \end{aligned}$$

where $a, b, c, d \in \mathbb{R}$ represent the chaotic bifurcation parameters.

---

### 2. Aizawa Attractor (3D Continuous)
The Aizawa attractor describes a three-dimensional autonomous system of ordinary differential equations (ODEs) exhibiting torus-shaped structural formation:

$$\begin{aligned} \frac{dx}{dt} &= (z - b) \cdot x - d \cdot y \\ \frac{dy}{dt} &= d \cdot x + (z - b) \cdot y \\ \frac{dz}{dt} &= c + a \cdot z - \frac{z^3}{3} - (x^2 + y^2)(1 + e \cdot z) + f \cdot z \cdot x^3 \end{aligned}$$

Typical parameter configuration: $a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1$. The numerical solution is computed by default using the 4th-order Runge-Kutta method (RK4).

---

### 3. Peter de Jong Attractor (2D Discrete)
A trigonometric multi-loop system exhibiting high sensitivity to phase shifts:

$$\begin{aligned} x_{n+1} &= \sin(a \cdot y_n) - \cos(b \cdot x_n) \\ y_{n+1} &= \sin(c \cdot x_n) - \cos(d \cdot y_n) \end{aligned}$$

---

## Architecture & Pipeline

The engine follows a strictly modular three-stage architecture:

```text
┌───────────────────────────┐
│     Dynamical Systems     │  --> GPU Tensor Initialization
│ (Parameter space, States) │
└─────────────┬─────────────┘
              │  Vectorized RK4 / Iteration (CUDA)
              ▼
┌───────────────────────────┐
│     Accumulation Buffer   │  --> Density Estimation & Log-Transformation
│ (2D/3D Tensor Histograms) │
└─────────────┬─────────────┘
              │  SSAA, ACES Tone Mapping & Color Grading
              ▼
┌───────────────────────────┐
│    Final Frame / Export   │  --> High-Res PNG / 4K Video Sequence
└───────────────────────────┘
