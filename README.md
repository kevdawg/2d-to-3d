# 2D to 3D Pipeline – Bas-Relief Generator

Transform 2D images into high-quality 3D models optimized for bas-relief CNC carving and 3D printing.
This pipeline uses AI-powered depth estimation (Marigold) to generate detailed depth maps, then converts them into STL, GLB, and OBJ files.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

---

## ⚡ Features

- 🎨 **AI Image Generation** — Gemini (FREE) or Imagen 3 (high quality)
- 🖼️ **Photo Enhancement** — AI upscaling using Real-ESRGAN
- ✨ **AI Image Repair** — Deblur and enhance with Imagen 3
- 🗑️ **Background Removal** — `rembg` or Remove.bg
- 🗺️ **Depth Map Generation** — Marigold (state-of-the-art)
- 🎯 **3D Model Creation** — Export STL, GLB, OBJ
- ⚙️ **Automatic Mesh Repair** — Clean and optimize using PyMeshLab
- 🧠 **Region-Aware Depth** — (Experimental) foreground/background separation
- 📊 **Quality Presets** — Ultra-Low → Ultra-High
- 🔄 **Batch Processing**
- ⏱️ **Performance Tracking**

---

# 🚀 Quick Start

## 1. Prerequisites

- **Python 3.10+**
- **Conda** (recommended)
- Windows / Linux / macOS
- **GPU recommended** (NVIDIA CUDA)

---

## 2. Clone Repository

```bash
git clone https://github.com/yourusername/2d-to-3d-pipeline.git
cd 2d-to-3d-pipeline
```

---

## 3. Set Up Environments

### **Automated (recommended)**

```bash
install.bat
```

### **Manual Setup**

```bash
# Image generation
conda create -n aigen python=3.10 -y
conda activate aigen
pip install -r environments/requirements_aigen.txt

# Marigold depth estimation
conda create -n marigold python=3.10 -y
conda activate marigold
pip install -r environments/requirements_marigold.txt

# 3D extrusion
conda create -n depth-to-3d python=3.10 -y
conda activate depth-to-3d
pip install -r environments/requirements_depth_to_3d.txt

# Photo preprocessing
conda create -n photo-prep python=3.10 -y
conda activate photo-prep
pip install -r environments/requirements_photo_prep.txt
```

> **Note:** Marigold downloads a ~2GB model on first run.

---

## 4. Set Up Credentials

### Gemini API (FREE)

Get key: https://aistudio.google.com/app/apikey

#### Windows:

```dos
setx GEMINI_API_KEY "your-api-key"
```

#### Linux/macOS:

```bash
echo 'export GEMINI_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### Remove.bg (optional, 50 free images)

```dos
setx REMOVEBG_API_KEY "your-api-key"
```

For Imagen 3, see **CREDENTIALS_SETUP.md**.

---

## 5. Run the Pipeline

### Windows:

```dos
pipeline\run_pipeline.bat
```

### Linux / macOS:

```bash
cd pipeline
conda activate photo-prep
python 2D_to_3D_pipeline.py
```

---

# 📸 Example Gallery

## Workflow Overview

**Input → Background Removal → Enhancement → Depth Map → 3D Model**

---

## Example 1: Calf Portrait

**Processing Time:** 4m 19s (Low Quality)

| Step | Preview | Details |
|------|---------|---------|
| 1 | Original Photo | 4032×3024 |
| 2 | Background Removed | Auto-cropped |
| 3 | AI Enhanced | Clarity boost |
| 4 | Depth Map | 512×477, 16-bit |
| 5 | 3D Model (STL) | 100mm × 10mm |

**Settings:**

```yaml
Quality: Low
Steps: 5
Ensemble: 1
Resolution: 512
Emboss: 0.2
Smoothing: 3
```

---

## Example 2: Oak Leaf (AI Generated)

**Processing Time:** 12m 34s (High Quality)

```yaml
Steps: 20
Ensemble: 5
Resolution: 1024
Emboss: 0.5
```

---

# 🎚️ Quality Comparison

### Low vs Medium vs High

| Preset | Time | Steps | Ensemble | Resolution | Use Case |
|--------|------|--------|----------|------------|----------|
| **Low** | ~4m | 5 | 1 | 512 | Fast preview |
| **Medium** ⭐ | ~12m | 10 | 3 | 768 | Recommended |
| **High** | ~30m | 20 | 5 | 1024 | Final detail |

---

# 🧱 Emboss Depth Comparison

| Emboss | Relief Depth | Description |
|--------|--------------|-------------|
| 0.1 | Very shallow | Coin-like |
| 0.2 | Low | Default |
| 0.3 ⭐ | Medium | Balanced |
| 0.5 | High | Wall art |
| 0.8 | Very deep | Sculptural |

---

# 🛠️ Use Cases

- **CNC Bas-Relief Carving**
- **3D Printing**
- **Medal/Coin Design**
- **Decorative Wall Art**

---

# 🖥️ Interactive Menu

```
============================================================
  2D to 3D Pipeline - Bas-Relief Generator
============================================================

1. Generate AI Image
2. Repair Blurry Photo (AI)
3. Transform 2D to 3D
4. Rerun Depth-to-Model
5. Edit Configuration
6. Quit
```

---

# 🔧 Command-Line Usage

### Generate AI Image

```bash
python scripts/image_generation/generate_with_gemini.py \
  --prompt "oak leaf" \
  --out data/AI_files/leaf.png
```

### Enhance Photo

```bash
python scripts/photo_preprocessing/ai_enhance.py \
  --input photo.jpg \
  --output enhanced.png \
  --method realesrgan
```

### Remove background

```bash
python scripts/photo_preprocessing/background_remover_removebg.py \
  --input photo.jpg \
  --output nobg.png
```

### Generate depth map

```bash
conda run -n marigold python scripts/depth_generation/marigold_cli.py \
  --input enhanced.png \
  --output depth.png \
  --steps 20 \
  --ensemble 5 \
  --processing_res 1024
```

### Create 3D model

```bash
conda run -n depth-to-3d python scripts/model_generation/extrude_cli.py \
  --input depth.png \
  --output model.stl \
  --width_mm 100 \
  --emboss 0.3 \
  --smoothing 3
```

---

# 📁 Project Structure

```text
2D-to-3D/
│
├── pipeline/
│   ├── 2D_to_3D_pipeline.py
│   ├── timing_tracker.py
│   ├── config.yaml
│   └── run_pipeline.bat
│
├── scripts/
│   ├── image_generation/
│   ├── photo_preprocessing/
│   ├── depth_generation/
│   └── model_generation/
│
├── environments/
├── models/
├── data/
├── examples/
├── install.bat
├── README.md
└── CREDENTIALS_SETUP.md
```

---

# ⚙️ Configuration

## Quality Presets

```yaml
marigold_presets:
  low_quality:
    marigold_steps: 5
    marigold_ensemble: 1
    marigold_processing_res: 512

  medium_quality:
    marigold_steps: 10
    marigold_ensemble: 3
    marigold_processing_res: 768

  high_quality:
    marigold_steps: 20
    marigold_ensemble: 5
    marigold_processing_res: 1024
```

## 3D Model Settings

```yaml
extrude_defaults:
  width_mm: 100.0
  max_height_mm: null
  smoothing: 3

  near_offset: "auto"
  far_offset: "auto"

  emboss: 0.2

  f_thic: 0.0
  f_near: -0.0
  f_back: 0.05

  vertex_colors: false
  scene_lights: false
  output_stl: true
  output_glb: false
  output_obj: false
```

---

# 📈 Performance Guide

### Typical Breakdown

```
Background Removal .... 1%
AI Enhancement ........ 3%
Depth Generation ..... 89%  ⚠️ Bottleneck
3D Extrusion .......... 5%
```

### Optimization Tips

- Use a **GPU**
- Lower resolution: 1024 → 768 → 512
- Lower ensemble: 5 → 3 → 1
- Fewer steps: 20 → 10

---

# 🖥️ Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|----------|-------------|---------|
| CPU | 4 cores | 8 cores | 16+ |
| RAM | 8 GB | 16 GB | 32+ |
| GPU | None | GTX 1660 | RTX 3060+ |
| Storage | 10 GB | 20 GB | 50 GB SSD |

---

# ❓ FAQ

**Q: Do I need a GPU?**  
A: No, but depth generation is 10–20× slower on CPU.

**Q: What image formats are supported?**  
Input: PNG, JPG, JPEG, BMP  
Output: PNG (depth), STL, GLB, OBJ

**Q: How do I deepen or shallow the relief?**  
Adjust `emboss` in `config.yaml`.

**Q: Depth map looks inverted?**  
Normal. White = near; black = far.

**Q: Can I use this commercially?**  
Pipeline = MIT. Marigold = Apache 2.0. Google APIs have ToS.

---

# 📜 License

- Pipeline code: **MIT**
- Marigold model: **Apache 2.0**
- See `LICENSE` for details.

---

# 🧬 Citation

```bibtex
@misc{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
  year={2023},
  eprint={2312.02145},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

```bibtex
@software{2d_to_3d_pipeline,
  title={2D to 3D Pipeline - Bas-Relief Generator},
  author={Kevin Burt},
  year={2025},
  url={https://github.com/kevdawg/2d-to-3d}
}
```

---

# 🙏 Acknowledgments

- Marigold Team – ETH Zürich
- Google – Gemini & Vertex AI
- Hugging Face – Diffusers
- Trimesh
- Remove.bg
- Real-ESRGAN

---

Made with ❤️ for makers, artists, and CNC enthusiasts.

