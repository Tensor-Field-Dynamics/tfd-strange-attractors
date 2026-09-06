<div align="center">
  <img src="https://raw.githubusercontent.com/Tensor-Field-Dynamics/.github/main/assets/tfd_banner.png" alt="Tensor Field Dynamics Banner" width="100%">
  
  # High-Performance Strange Attractor Simulation Engine
  
  [![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Optimized-EE4C2C.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)
  [![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

  <p align="center">
    <b>GPU-accelerated numerical integration and rendering pipeline for deterministic-chaotic dynamical systems, 3D projective camera geometry, and topological data art.</b>
  </p>
</div>

---

## Table of Contents
- [Overview](#overview)
- [Architecture & Pipeline](#architecture--pipeline)
- [Mathematical Systems](#mathematical-systems)
  - [Discrete Attractors (2D)](#discrete-attractors-2d)
  - [Continuous Attractors (3D ODEs)](#continuous-attractors-3d-odes)
- [Camera & 3D Projective Pipeline](#camera--3d-projective-pipeline)
- [Color Grading, Themes & Tonemapping](#color-grading-themes--tonemapping)
- [Branding & Watermarking Engine](#branding--watermarking-engine)
- [Render State Management](#render-state-management)
- [Installation](#installation)
- [Quickstart & Usage](#quickstart--usage)
- [Directory Structure](#directory-structure)
- [License](#license)

---

## Overview

This module provides a production-grade, GPU-accelerated numerical integration and rendering engine to simulate, project, and visualize chaotic phase spaces of continuous and discrete dynamical systems.

### Core Features
* **Fully Vectorized GPU Integration:** Trajectory computations and numerical integrations are executed via native PyTorch CUDA tensors with batched memory allocation (`batch_size = 2_000_000`) for high-throughput VRAM utilization without CPU transfer bottlenecks.
* **3D Virtual Camera & Perspective Projection Matrix:** Arbitrary 6-DOF camera placement (`CameraConfig`) with `look_at`, configurable FOV, near/far planes, View-Projection matrix transformations (`VP = Proj @ View`), and NDC-to-screen pixel rasterization.
* **Transient Chaos Filtering (Warm-up / Burn-in Phase):** Dynamically filters out unstable initial boundary conditions and transient states before accumulation, isolating clean strange attractor manifolds.
* **High-End Volumetric Shading & Tone Mapping:** Logarithmic density accumulation buffer, 2x–4x Supersampling Anti-Aliasing (SSAA), custom ACES (Academy Color Encoding System) filmic tone mapping curve, and multi-glow color themes.
* **Integrated Branding & Overlay Compositing:** Modular watermarking pipeline with Lanczos resampling, alpha composite blending, custom opacity, scaling, and margin configurations.
* **State Serialization:** Reproducible experiment tracking and persistent render state management via `RenderState` and `RenderConfigManager`.

---

## Architecture & Pipeline

The engine follows a strictly decoupled, high-performance execution pipeline:

```text
┌──────────────────────────────────────────────────────────┐
│              Dynamical Systems & Phase Space             │
│   (Discrete Maps / Continuous ODEs: Euler & RK4 Integr.) │
└────────────────────────────┬─────────────────────────────┘
                             │  Vectorized PyTorch / CUDA Batches
                             ▼
┌──────────────────────────────────────────────────────────┐
│              3D Camera & Projection Pipeline             │
│   (Look-At View Matrix, Perspective Proj, NDC Clipping)  │
└────────────────────────────┬─────────────────────────────┘
                             │  Screen-Space Rasterization
                             ▼
┌──────────────────────────────────────────────────────────┐
│              Histogram Accumulation Buffer               │
│   (Atomic Index-Put Density Mapping & Log-Transform)     │
└────────────────────────────┬─────────────────────────────┘
                             │  SSAA, ACES Filmic Tone Mapping, Color Interpolation
                             ▼
┌──────────────────────────────────────────────────────────┐
│           Post-Processing & Watermark Engine             │
│   (Alpha Blending, Lanczos Downsampling, Branding)       │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                     Export Artifacts                     │
│   (High-Res PNG, Print Posters up to 8K / A1, MP4 Reels) │
└──────────────────────────────────────────────────────────┘
```

---

## Mathematical Systems

The engine implements 12 chaotic dynamical systems spanning discrete recursive maps and continuous nonlinear differential equations:

### Discrete Attractors (2D)

#### 1. Clifford Attractor

Defined by the iterative recurrence relations:

$$
\begin{aligned}
x_{n+1} &= \sin(a \cdot y_n) + c \cdot \cos(a \cdot x_n) \\
y_{n+1} &= \sin(b \cdot x_n) + d \cdot \cos(b \cdot y_n)
\end{aligned}
$$

*Default parameters:* $a = -1.4, b = 1.6, c = 1.0, d = 0.7$.

#### 2. Peter de Jong Attractor

Multi-loop trigonometric attractor with high sensitivity to phase perturbations:

$$
\begin{aligned}
x_{n+1} &= \sin(a \cdot y_n) - \cos(b \cdot x_n) \\
y_{n+1} &= \sin(c \cdot x_n) - \cos(d \cdot y_n)
\end{aligned}
$$

---

### Continuous Attractors (3D ODEs)

#### 3. Thomas Attractor (Cyclically Symmetric)

Cyclically symmetric system modeling friction in a 3D velocity field:

$$
\frac{dx}{dt} = \sin(y) - b \cdot x, \quad \frac{dy}{dt} = \sin(z) - b \cdot y, \quad \frac{dz}{dt} = \sin(x) - b \cdot z
$$

*Default parameters:* $b = 0.19, dt = 0.05$.

#### 4. Aizawa Attractor

Continuous autonomous system exhibiting torus-shaped structural formation and sphere-like core:

$$
\begin{aligned}
\frac{dx}{dt} &= (z - b) \cdot x - d \cdot y \\
\frac{dy}{dt} &= d \cdot x + (z - b) \cdot y \\
\frac{dz}{dt} &= c + a \cdot z - \frac{z^3}{3} - (x^2 + y^2)(1 + e \cdot z) + f \cdot z \cdot x^3
\end{aligned}
$$

*Default parameters:* $a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1, dt = 0.01$.

#### 5. Lorenz Attractor

The canonical atmospheric convection model:

$$
\frac{dx}{dt} = \sigma (y - x), \quad \frac{dy}{dt} = x (\rho - z) - y, \quad \frac{dz}{dt} = x \cdot y - \beta \cdot z
$$

*Default parameters:* $\sigma = 10.0, \rho = 28.0, \beta = 8/3, dt = 0.005$.

#### 6. Dadras Attractor

Five-parameter continuous chaotic system exhibiting intricate multi-scroll topologies:

$$
\begin{aligned}
\frac{dx}{dt} &= y - a \cdot x + b \cdot y \cdot z \\
\frac{dy}{dt} &= c \cdot y - x \cdot z + z \\
\frac{dz}{dt} &= d \cdot x \cdot y - e \cdot z
\end{aligned}
$$

*Default parameters:* $a = 3.0, b = 2.7, c = 1.7, d = 2.0, e = 9.0, dt = 0.005$.

#### 7. Chen Attractor

Dual-scroll chaotic attractor belonging to the generalized Lorenz family:

$$
\frac{dx}{dt} = a(y - x), \quad \frac{dy}{dt} = (c - a)x - x \cdot z + c \cdot y, \quad \frac{dz}{dt} = x \cdot y - b \cdot z
$$

*Default parameters:* $a = 35.0, b = 3.0, c = 28.0, dt = 0.002$.

#### 8. Lorenz-83 Attractor

Simplified model of the atmospheric general circulation:

$$
\begin{aligned}
\frac{dx}{dt} &= -a \cdot x - y^2 - z^2 + a \cdot F \\
\frac{dy}{dt} &= -y + x \cdot y - b \cdot x \cdot z + G \\
\frac{dz}{dt} &= -z + b \cdot x \cdot y + x \cdot z
\end{aligned}
$$

*Default parameters:* $a = 0.95, b = 7.91, F = 4.83, G = 4.66, dt = 0.002$.

#### 9. Rössler Attractor

Continuous ODE system designed for studying spiral chaos:

$$
\frac{dx}{dt} = -y - z, \quad \frac{dy}{dt} = x + a \cdot y, \quad \frac{dz}{dt} = b + z(x - c)
$$

*Default parameters:* $a = 0.2, b = 0.2, c = 5.7, dt = 0.01$.

#### 10. Halvorsen Attractor

Cyclically symmetric three-dimensional attractor with wide orbits:

$$
\begin{aligned}
\frac{dx}{dt} &= -a \cdot x - 4y - 4z - y^2 \\
\frac{dy}{dt} &= -a \cdot y - 4z - 4x - z^2 \\
\frac{dz}{dt} &= -a \cdot z - 4x - 4y - x^2
\end{aligned}
$$

*Default parameters:* $a = 1.89, dt = 0.005$.

#### 11. Rabinovich–Fabrikant Attractor

Nonlinear dynamical system describing modulational instability in non-equilibrium media:

$$
\begin{aligned}
\frac{dx}{dt} &= y(z - 1 + x^2) + \gamma \cdot x \\
\frac{dy}{dt} &= x(3z + 1 - x^2) + \gamma \cdot y \\
\frac{dz}{dt} &= -2z(\alpha + x \cdot y)
\end{aligned}
$$

*Default parameters:* $\alpha = 0.14, \gamma = 0.10, dt = 0.005$.

#### 12. Three-Scroll Unified Attractor

Multi-scroll continuous chaotic attractor with complex topological folding:

$$
\begin{aligned}
\frac{dx}{dt} &= a(y - x) + d \cdot x \cdot z \\
\frac{dy}{dt} &= b \cdot x - x \cdot z + f \cdot y \\
\frac{dz}{dt} &= c \cdot z + x \cdot y - e \cdot x^2
\end{aligned}
$$

*Default parameters:* $a = 32.48, b = 45.84, c = 1.18, d = 0.13, e = 0.57, f = 14.7, dt = 0.001$.

---

## Camera & 3D Projective Pipeline

The engine includes a full virtual camera system in `core/camera.py`:

```python
from core.camera import CameraConfig, CAMERA_PRESETS, project_points

# Use pre-configured camera viewports:
# - "Three-Quarter-Prospect" (default for 3D continuous attractors)
# - "Side Profile"
# - "Macro Detail"
# - "Default 2D" (for Clifford and Peter de Jong)
camera = CAMERA_PRESETS["Three-Quarter-Prospect"]

# Or define custom 6-DOF camera positions:
custom_cam = CameraConfig(
    position=(15.0, 10.0, 15.0),
    look_at=(0.0, 0.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fov_degrees=45.0,
    near=0.1,
    far=1000.0
)
```

The camera applies homogeneous transformation matrices directly in PyTorch GPU tensors:

$$
\mathbf{P}_{\text{clip}} = \mathbf{M}_{\text{proj}} \cdot \mathbf{M}_{\text{view}} \cdot \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}, \quad \mathbf{P}_{\text{ndc}} = \frac{\mathbf{P}_{\text{clip}}^{xyz}}{w_{\text{clip}}}
$$

---

## Color Grading, Themes & Tonemapping

Post-processing operates on the accumulated density matrix using ACES filmic tone mapping:

$$
f(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}
$$

12 color profiles are available out of the box in `config/themes.py`:
- `neon_cyberpunk`, `dark_matter`, `solar_flare`, `emerald_forest`, `deep_ocean`, `amethyst_nebula`, `crimson_synth`, `golden_hour`, `arctic_aurora`, `volcanic_ash`, `cyber_ghost`, `cosmic_twilight`.

---

## Branding & Watermarking Engine

The module `core/shading.py` and `config/branding.py` provide asset management and compositing for logos and watermarks:

```python
from core.shading import apply_watermark
from config.branding import BRANDING_ASSETS

# Apply watermark with alpha blending & Lanczos downsampling
watermarked_image = apply_watermark(
    frame=image,
    watermark_path=BRANDING_ASSETS["white_logo"],
    enable_watermark=True,
    opacity=0.85,
    scale=0.15,
    margin=50
)
```

Available assets in `assets/branding/`:
- `tfd_logo_white.png` (Default)
- `tfd_logo_horizontal_1.png` / `tfd_logo_horizontal_2.png`
- `tfd_logo_1.png` / `tfd_logo_2.png`

---

## Render State Management

`config/state_manager.py` enables full serialization and deserialization of rendering parameters for reproducible generative art workflows:

```python
from config.state_manager import RenderState, RenderConfigManager

state = RenderState(
    system_type="HalvorsenAttractor",
    system_parameters={"a": 1.89, "dt": 0.005},
    numerical_settings={"num_points": 5_000_000, "iters_per_point": 50},
    camera_settings={"preset": "Three-Quarter-Prospect"},
    pipeline_settings={"theme": "dark_matter", "resolution": "insta_portrait"}
)

RenderConfigManager.save_config(state, "outputs/configs/halvorsen_run.json")
loaded_state = RenderConfigManager.load_config("outputs/configs/halvorsen_run.json")
```

---

## Installation

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ support (recommended for multi-million point renders)

```bash
git clone https://github.com/Tensor-Field-Dynamics/tfd-strange-attractors.git
cd tfd-strange-attractors
pip install -r requirements.txt
```

---

## Quickstart & Usage

### 1. Render Still Image
```python
from core.attractor_engine import HalvorsenAttractor
from core.camera import CAMERA_PRESETS
from core.shading import render_image, apply_watermark
from config.resolutions import TARGET_CONFIGS, TargetPlatform
from config.themes import THEMES
from config.branding import BRANDING_ASSETS
from PIL import Image

# 1. Initialize Attractor & Resolution
config = TARGET_CONFIGS[TargetPlatform.WALLPAPER_4K]
attractor = HalvorsenAttractor(device="cuda")

# 2. Compute Density Map via 3D Camera Projection
density = attractor.generate_density_map(
    width=config.internal_width,
    height=config.internal_height,
    num_points=10_000_000,
    iters_per_point=50,
    camera_config=CAMERA_PRESETS["Three-Quarter-Prospect"]
)

# 3. Tone Mapping & Post-Processing
image = render_image(density, THEMES["dark_matter"])
if config.ssaa_factor > 1:
    image = image.resize((config.width, config.height), Image.Resampling.LANCZOS)

# 4. Apply Watermark & Save
image = apply_watermark(image, BRANDING_ASSETS["white_logo"])
image.save("halvorsen_4k.png")
```

### 2. Generate Buildup Animation
```bash
python scripts/generate_buildup.py
```

### 3. Generate Theme Showcase
```bash
python scripts/generate_theme_showcase.py
```

---

## Directory Structure

```text
tfd-strange-attractors/
├── assets/
│   └── branding/              # Vector-derived high-resolution watermark assets
├── config/
│   ├── branding.py            # Branding asset enumerations and aliases
│   ├── resolutions.py         # Social, Wallpaper & DIN print resolution configurations
│   ├── state_manager.py       # RenderState JSON serialization & persistence
│   └── themes.py              # 12 curated color palettes and glow profiles
├── core/
│   ├── attractor_engine.py    # 12 GPU-accelerated attractor implementations
│   ├── camera.py              # 3D View & Perspective projection camera system
│   └── shading.py             # ACES tone mapping, bloom & watermark compositing
├── scripts/
│   ├── generate_art.py        # Single image generative script
│   ├── generate_buildup.py    # Multi-frame iterative build-up animation
│   ├── generate_reel.py       # Dynamic rotating / flying camera reels
│   ├── generate_theme_showcase.py  # Automated 12-theme showcase generator
│   └── compile_video.py       # Frame sequence to MP4/ProRes encoding
├── requirements.txt           # Project dependencies
├── LICENSE.md                 # MIT License
└── README.md                  # Technical documentation
```

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
