# 2D to 3D Pipeline - Bas-Relief Generator

Transform 2D images into high-quality 3D models optimized for bas-relief CNC carving and 3D printing. This pipeline uses AI-powered depth estimation (Marigold) to generate detailed depth maps, then converts them into STL, GLB, and OBJ files.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- 🎨 **AI Image Generation** - Generate images with Gemini (FREE) or Imagen 3 (high quality)
- 🖼️ **Photo Enhancement** - Automatic AI upscaling and clarity enhancement
- 🗑️ **Background Removal** - Remove.bg API integration with auto-cropping
- 🗺️ **Depth Map Generation** - State-of-the-art Marigold depth estimation
- 🎯 **3D Model Creation** - Export to STL, GLB, and OBJ formats
- ⚙️ **Quality Presets** - Low, medium, and high quality options
- 🔄 **Batch Processing** - Process multiple images automatically
- 📊 **Performance Tracking** - Detailed timing and bottleneck identification

---

## Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Conda** (recommended for environment management)
- **Windows**, **Linux**, or **macOS**
- **GPU recommended** (NVIDIA with CUDA for faster processing)

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/2d-to-3d-pipeline.git
cd 2d-to-3d-pipeline
```

### 3. Set Up Environments

Run the automated installer:

```cmd
install.bat
```

Or manually create environments:

```bash
# Image generation environment
conda create -n imagen python=3.10 -y
conda activate imagen
pip install -r environments/requirements_imagen.txt

# Marigold depth estimation environment
conda create -n marigold python=3.10 -y
conda activate marigold
pip install -r environments/requirements_marigold.txt

# 3D extrusion environment
conda create -n depth-to-3d python=3.10 -y
conda activate depth-to-3d
pip install -r environments/requirements_depth_to_3d.txt

# Photo preprocessing environment
conda create -n photo-prep python=3.10 -y
conda activate photo-prep
pip install -r environments/requirements_photo_prep.txt
```

**Note:** The Marigold model (~2GB) will download automatically on first use.

### 4. Set Up Credentials

**For basic usage (FREE):**
```cmd
# Get API key from: https://aistudio.google.com/app/apikey

# Windows - Set permanently (recommended)
setx GEMINI_API_KEY "your-api-key-here"

# Linux/Mac - Add to ~/.bashrc or ~/.zshrc
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**For background removal (optional):**
```cmd
# Get API key from: https://www.remove.bg/users/sign_up
# 50 free images/month, then $0.20 per image

setx REMOVEBG_API_KEY "your-api-key-here"
```

**For high-quality images (requires billing):**
See [`CREDENTIALS_SETUP.md`](CREDENTIALS_SETUP.md) for detailed Vertex AI setup instructions.

### 5. Run the Pipeline

```cmd
# Windows
pipeline\run_pipeline.bat

# Linux/Mac
cd pipeline
conda activate photo-prep
python 2D_to_3D_pipeline.py
```

---

## Example Gallery

### Workflow Overview

```
Input Image → Background Removal → AI Enhancement → Depth Map → 3D Model
   (JPG)          (transparent)      (upscaled)      (16-bit)     (STL)
```

### Example 1: Calf Portrait (Photo Source)

| Step | Preview | Details |
|------|---------|---------|
| **1. Original Photo** | ![Original](examples/calf_original.jpg) | 4032×3024 (12.2 MP)<br>Smartphone photo |
| **2. Background Removed** | ![No BG](examples/calf_nobg.png) | 2091×2242 (4.7 MP)<br>Auto-cropped 48% |
| **3. AI Enhanced** | ![Enhanced](examples/calf_enhanced.png) | 2091×2242<br>Clarity + detail boost |
| **4. Depth Map** | ![Depth](examples/calf_depth.png) | 512×477 (16-bit)<br>Grayscale depth |
| **5. 3D Model** | ![3D Model](examples/calf_3d.png) | STL export<br>100mm × 10mm relief |

**Processing Time:** 4m 19s (Low Quality preset)
- Background removal: 2.3s (1%)
- AI enhancement: 8.7s (3%)
- Depth generation: 231s (89%) ⚠️ Bottleneck
- 3D extrusion: 14s (5%)

**Settings Used:**
```yaml
Quality: Low
Steps: 5 | Ensemble: 1 | Resolution: 512
Emboss: 0.2 | Smoothing: 3
```

---

### Example 2: Oak Leaf (AI Generated)

| Step | Preview | Details |
|------|---------|---------|
| **1. AI Generated** | ![Generated](examples/leaf_generated.png) | Gemini Flash<br>1024×1024 |
| **2. Depth Map** | ![Depth](examples/leaf_depth.png) | High quality preset<br>1024×953 |
| **3. 3D Model (Front)** | ![Front](examples/leaf_3d_front.png) | Dramatic emboss<br>emboss=0.5 |
| **4. 3D Model (Side)** | ![Side](examples/leaf_3d_side.png) | Shows relief depth<br>~8mm height |

**Processing Time:** 12m 34s (High Quality preset)

**Settings Used:**
```yaml
Quality: High
Steps: 20 | Ensemble: 5 | Resolution: 1024
Emboss: 0.5 | Smoothing: 3
```

---

### Quality Comparison: Low vs Medium vs High

**Same Input Image - Different Quality Settings**

| Preset | Preview | Time | Steps | Ensemble | Resolution | Use Case |
|--------|---------|------|-------|----------|------------|----------|
| **Low** | ![Low](examples/compare_low.png) | ~4m | 5 | 1 | 512 | Fast preview, testing |
| **Medium** | ![Med](examples/compare_med.png) | ~12m | 10 | 3 | 768 | Balanced production |
| **High** | ![High](examples/compare_high.png) | ~30m | 20 | 5 | 1024 | Final output, detail work |

**Visible Differences:**
- **Low**: Faster, slight noise in smooth areas, good for prototyping
- **Medium**: Clean depth, good detail, recommended for most projects ⭐
- **High**: Maximum detail, smoothest gradients, best for large prints

---

### Emboss Depth Comparison

**Same Depth Map - Different Emboss Settings**

| Emboss | Preview | Relief Depth | Description |
|--------|---------|--------------|-------------|
| **0.1** | ![Subtle](examples/emboss_01.png) | Very shallow | Subtle texture, coin-like |
| **0.2** | ![Low](examples/emboss_02.png) | Low relief | Current low preset |
| **0.3** | ![Standard](examples/emboss_03.png) | Medium relief | Balanced, versatile ⭐ |
| **0.5** | ![Dramatic](examples/emboss_05.png) | High relief | Dramatic depth, wall art |
| **0.8** | ![Extreme](examples/emboss_08.png) | Very deep | Maximum depth, sculpture |

---

### Use Cases

#### CNC Bas-Relief Carving
![CNC Example](examples/cnc_carved.jpg)
- **Material**: Hardwood maple
- **Size**: 150mm × 150mm × 15mm
- **Settings**: Medium quality, emboss=0.3
- **Carving time**: 2.5 hours

#### 3D Printing
![3D Print](examples/3d_printed.jpg)
- **Material**: PLA filament
- **Size**: 100mm diameter
- **Settings**: High quality, emboss=0.4, smoothing=5
- **Print time**: 6 hours

#### Medal/Coin Design
![Medal](examples/medal.jpg)
- **Application**: Custom award medal
- **Settings**: High quality, emboss=0.15 (shallow)
- **Process**: STL → CNC milling → casting

---

## Usage

### Interactive Menu

The pipeline provides an interactive menu:

```
============================================================
  2D to 3D Pipeline - Bas-Relief Generator
============================================================

MAIN MENU:
------------------------------------------------------------
  1. Generate AI Image
  2. Transform 2D to 3D
  3. Rerun Depth-to-Model (new settings)
  4. Edit Configuration
  5. Quit
------------------------------------------------------------
```

#### Menu 1: Generate AI Image
- **Gemini (FREE)**: Basic quality, 15 req/min limit
- **Imagen ($0.01)**: High quality, requires Google Cloud billing

#### Menu 2: Transform 2D to 3D
- Select quality preset: Low (~4m), Medium (~12m), High (~30m)
- Choose source: AI_files/ or Photos/ (auto-enhances photos)
- Batch process entire folder or single image

#### Menu 3: Rerun Depth-to-Model
- Keep same depth map, try different 3D extrusion settings
- Experiment with emboss, smoothing, frame styles
- Fast iteration without regenerating depth

#### Menu 4: Edit Configuration
- Opens `config.yaml` in default text editor
- Modify quality presets, enhancement settings
- Change default parameters

---

### Command-Line Usage

For automation or scripting:

```bash
# Generate image
python scripts/image_generation/generate_with_gemini.py \
  --prompt "oak leaf" \
  --out data/AI_files/leaf.png

# Enhance photo (optional)
python scripts/photo_preprocessing/ai_enhance.py \
  --input photo.jpg \
  --output enhanced.png \
  --method realesrgan \
  --scale 2

# Remove background (optional)
python scripts/photo_preprocessing/background_remover_removebg.py \
  --input photo.jpg \
  --output nobg.png \
  --method removebg

# Generate depth map
conda run -n marigold python scripts/depth_generation/marigold_cli.py \
  --input enhanced.png \
  --output depth.png \
  --steps 20 \
  --ensemble 5 \
  --processing_res 1024

# Create 3D model
conda run -n depth-to-3d python scripts/model_generation/extrude_cli.py \
  --input depth.png \
  --output model.stl \
  --width_mm 100 \
  --emboss 0.3 \
  --smoothing 3
```

---

## Project Structure

```
2D-to-3D/
│
├── pipeline/
│   ├── 2D_to_3D_pipeline.py        # Main interactive pipeline
│   ├── timing_tracker.py           # Timing and progress display
│   ├── config.yaml                 # Configuration settings
│   └── run_pipeline.bat            # Windows launcher
│
├── scripts/
│   ├── image_generation/
│   │   ├── generate_with_gemini.py      # FREE tier (Gemini Flash)
│   │   └── generate_with_imagen3.py     # High quality (Imagen 3)
│   │
│   ├── photo_preprocessing/
│   │   ├── ai_enhance.py                # AI upscaling & enhancement
│   │   ├── background_remover_removebg.py # Background removal
│   │   └── depth_masking.py             # Mask depth maps
│   │
│   ├── depth_generation/
│   │   └── marigold_cli.py              # Depth map generation
│   │
│   └── model_generation/
│       ├── extrude_cli.py               # CLI wrapper
│       └── extrude.py                   # 3D extrusion logic
│
├── environments/
│   ├── requirements_imagen.txt
│   ├── requirements_marigold.txt
│   ├── requirements_depth_to_3d.txt
│   └── requirements_photo_prep.txt
│
├── models/
│   ├── marigold_model/              # Auto-downloaded (git ignored)
│   └── RealESRGAN/                  # Auto-downloaded (git ignored)
│
├── data/
│   ├── AI_files/                    # AI-generated images
│   ├── Photos/                      # User photos
│   ├── Photos_enhanced/             # Enhanced photos
│   └── 3D_files/                    # Output 3D models
│
├── examples/                        # Gallery images
├── install.bat                      # Automated setup
├── .gitignore
├── README.md
└── CREDENTIALS_SETUP.md
```

---

## Configuration

Edit `pipeline/config.yaml` to customize:

### Quality Presets

```yaml
marigold_presets:
  low_quality:
    marigold_steps: 5           # Inference steps (3-30)
    marigold_ensemble: 1        # Ensemble size (1-15)
    marigold_processing_res: 512 # Internal resolution (384-1536)
    
  medium_quality:
    marigold_steps: 10
    marigold_ensemble: 3
    marigold_processing_res: 768
    
  high_quality:
    marigold_steps: 20
    marigold_ensemble: 5
    marigold_processing_res: 1024
```

### 3D Model Settings

```yaml
extrude_defaults:
  width_mm: 100.0           # Model width (10-1000mm)
  max_height_mm: 10.0       # Relief depth (1-100mm)
  smoothing: 3              # Median filter (odd: 1,3,5,7,9)
  
  # Depth mapping
  near_offset: 0.0          # Near clipping (0.0-1.0)
  far_offset: 1.0           # Far clipping (0.0-1.0)
  
  # Relief depth
  emboss: 0.3               # Emboss factor (0.1-0.8)
  
  # Frame/border
  f_thic: 0.05              # Frame thickness (0-0.2)
  f_near: -0.15             # Frame position (-0.5-0)
  f_back: 0.01              # Back thickness (0-0.1)
  
  # Output options
  vertex_colors: true       # Include RGB colors
  scene_lights: true        # Add lights to GLB
  prepare_for_3d_printing: false
  zip_outputs: false
```

### Photo Enhancement

```yaml
ai_enhancement:
  method: "realesrgan"      # "realesrgan" or "lanczos"
  upscale_factor: 2         # 1x, 2x, or 4x
  target_resolution: 1024   # Target size for depth
  max_input_size: 2048      # Max before auto-fallback
  
  clarity_strength: 1.3     # Clarity boost (0.5-2.0)
  detail_amount: 1.2        # Detail enhancement (0.5-2.0)
  sharpen_strength: 150     # Sharpening % (50-300)
```

See comments in `config.yaml` for detailed parameter descriptions.

---

## Performance Guide

### Timing Expectations

**Low Quality** (~4 minutes)
- Best for: Testing, previews, iteration
- Output: Acceptable for small prints (<50mm)

**Medium Quality** (~12 minutes) ⭐ Recommended
- Best for: Production work, balanced results
- Output: Clean depth, good detail

**High Quality** (~30 minutes)
- Best for: Final output, large prints, fine detail
- Output: Maximum quality, smoothest gradients

### Bottleneck Analysis

From typical output:
```
Step Breakdown:
├─ [1/4] Background Removal .......... 2.3s   (1%)
├─ [2/4] AI Enhancement .............. 8.7s   (3%)
├─ [3/4] Depth Generation ........... 231.0s  (89%) ⚠️ BOTTLENECK
└─ [4/4] 3D Extrusion ............... 14.0s   (5%)
```

**Optimization tips:**
1. **GPU required** - Marigold is 10-20× faster on NVIDIA GPU
2. **Reduce resolution** - 768 instead of 1024 saves ~40% time
3. **Lower ensemble** - 3 instead of 5 saves ~35% time
4. **Fewer steps** - 10 instead of 20 saves ~45% time

### Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 8 GB | 16 GB | 32+ GB |
| **GPU** | None (CPU) | GTX 1660 | RTX 3060+ |
| **Storage** | 10 GB | 20 GB | 50+ GB SSD |

**GPU impact on timing:**
- CPU only: ~8-10× slower
- GTX 1660: ~3× slower  
- RTX 3060: 1× (baseline)
- RTX 4090: ~2× faster

---

## Troubleshooting

### "GEMINI_API_KEY not set"
- Set environment variable: `setx GEMINI_API_KEY "your-key"`
- Restart terminal/command prompt
- Verify: `echo %GEMINI_API_KEY%` (Windows) or `echo $GEMINI_API_KEY` (Linux)

### "Could not find conda"
- Ensure Conda installed and added to PATH
- Specify full path in `config.yaml`: `conda_exe: "C:/path/to/conda.exe"`
- Test: `conda --version`

### "Model not found"
- Marigold downloads automatically on first use (~2GB, one-time)
- Check internet connection
- Verify ~2GB free disk space in `models/marigold_model/`

### "Permission denied" (Vertex AI)
- Enable API: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com
- Set up billing: https://console.cloud.google.com/billing
- See [CREDENTIALS_SETUP.md](CREDENTIALS_SETUP.md)

### "Out of memory" during depth generation
- Reduce `marigold_processing_res` (1024 → 768 → 512)
- Lower `marigold_ensemble` (5 → 3 → 1)
- Close other applications
- Consider GPU upgrade

### Poor depth map quality
- Use AI enhancement before depth generation
- Try high quality preset
- Ensure input image has clear details and edges
- Remove busy backgrounds (use background removal)

### 3D model looks flat
- Increase `emboss` (0.3 → 0.5 → 0.8)
- Adjust `far_offset` (1.0 → 0.7) to remove distant features
- Check depth map visually - white=near, black=far

### Artifacts in 3D model
- Increase `smoothing` (3 → 5 → 7)
- Increase `marigold_ensemble` (1 → 3 → 5) for stability
- Enable mesh repair in config (medium/high quality)

---

## API Costs

### Gemini API (FREE Tier)
- **Cost**: $0.00
- **Limits**: 15 requests/minute, 1,500 requests/day
- **Quality**: Good for testing and personal projects
- **Get key**: https://aistudio.google.com/app/apikey

### Remove.bg (Background Removal)
- **Cost**: $0.00 for first 50 images/month, then $0.20/image
- **Quality**: Best-in-class background removal
- **Get key**: https://www.remove.bg/users/sign_up

### Vertex AI - Imagen 3 (Paid)
- **Cost**: ~$0.01-0.04 per 1024×1024 image
- **Limits**: Pay-as-you-go, high limits
- **Quality**: Excellent, production-ready
- **Setup**: Requires Google Cloud billing account

### Marigold (Offline)
- **Cost**: $0.00 (runs locally)
- **Requirements**: Downloaded model (~2GB, one-time)
- **Performance**: GPU strongly recommended (10-20× faster)

---

## Advanced Usage

### Testing Different Settings

Quick config changes for experimentation:

```yaml
# Fast preview (2-3 minutes)
test_fast:
  marigold_steps: 3
  marigold_ensemble: 1
  marigold_processing_res: 384
  emboss: 0.2

# Dramatic relief (strong depth)
test_dramatic:
  marigold_steps: 10
  marigold_ensemble: 3
  marigold_processing_res: 768
  emboss: 0.6
  far_offset: 0.7

# Ultra-smooth (best for printing)
test_smooth:
  marigold_steps: 20
  marigold_ensemble: 5
  marigold_processing_res: 1024
  emboss: 0.3
  smoothing: 7
```

### Batch Processing

Process entire folders automatically:

```python
# In pipeline menu, select:
# 2. Transform 2D to 3D
# 4. Batch Process Folder
# Select quality preset
# All images in selected folder will be processed
```

**Batch output:**
```
================================================================
  BATCH PROGRESS: 2/5 images (40%)
================================================================
  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

  Statistics:
  ├─ Completed: 2 images (avg 4m 15s each)
  ├─ Remaining: 3 images
  └─ Estimated time remaining: 12m 45s
================================================================
```

### Custom Workflows

#### Workflow 1: High-Detail Small Objects
*Coins, medals, jewelry*

```yaml
marigold_steps: 20
marigold_ensemble: 5
marigold_processing_res: 1024
emboss: 0.15              # Shallow relief
smoothing: 3              # Preserve detail
width_mm: 50              # Small size
```

#### Workflow 2: Large Wall Art
*Portraits, landscapes, decorative panels*

```yaml
marigold_steps: 15
marigold_ensemble: 3
marigold_processing_res: 1024
emboss: 0.5               # Deep relief
smoothing: 5              # Smooth for viewing distance
width_mm: 300             # Large size
```

#### Workflow 3: 3D Printing Miniatures
*Game pieces, figurines*

```yaml
marigold_steps: 20
marigold_ensemble: 5
marigold_processing_res: 1024
emboss: 0.4
smoothing: 7              # Very smooth for printing
prepare_for_3d_printing: true
```

---

## Output Files

For each processed image, the pipeline generates:

```
3D_files/
└── calf_test_low_quality/
    ├── calf_test.jpg                    # Original image (copied)
    ├── calf_test_nobg.png              # Background removed
    ├── calf_test_ai_enhanced.png       # AI enhanced version
    ├── calf_test_depth_16bit.png       # 16-bit depth map (grayscale)
    ├── calf_test.stl                    # STL for 3D printing/CNC
    ├── calf_test.glb                    # GLB with colors and lights (if enabled)
    ├── calf_test.obj                    # OBJ for editing (if enabled)
    └── commands.txt                     # Command log for reproducibility
```

### File Descriptions

**Depth Map** (`*_depth_16bit.png`)
- 16-bit grayscale PNG
- White = near (high points)
- Black = far (low points)
- Used as input for 3D extrusion

**STL** (`*.stl`)
- Binary STL format
- Ready for CNC machining or 3D printing
- Compatible with all slicers and CAM software

**GLB** (`*.glb`)
- Compact 3D format with embedded textures
- Includes vertex colors from original image
- Pre-lit for web viewing (three.js, Babylon.js)

**OBJ** (`*.obj`)
- Text-based 3D format
- Best for editing in Blender, MeshLab, etc.
- Separate material/texture files

**Commands Log** (`commands.txt`)
- Complete command history
- Allows exact reproduction of results
- Useful for automation and scripting

---

## Testing & Optimization Guide

### Parameter Testing Matrix

For systematic quality/performance testing, try these combinations:

#### Test Set 1: Speed vs Quality Baseline
*Establish timing curves for your hardware*

| Test | Steps | Ensemble | Res | Expected Time |
|------|-------|----------|-----|---------------|
| Ultra-Fast | 3 | 1 | 384 | ~2m |
| Fast | 5 | 1 | 512 | ~4m |
| Balanced | 10 | 3 | 768 | ~12m |
| High | 20 | 5 | 1024 | ~30m |

#### Test Set 2: Emboss Depth
*Find optimal relief depth for your application*

| Emboss | Far Offset | Relief Type |
|--------|------------|-------------|
| 0.1 | 1.0 | Very subtle (coin) |
| 0.2 | 1.0 | Low relief |
| 0.3 | 1.0 | Medium relief ⭐ |
| 0.5 | 0.8 | High relief (wall art) |
| 0.8 | 0.6 | Very deep (sculpture) |

#### Test Set 3: Smoothing vs Detail
*Balance between surface detail and printability*

| Smoothing | Best For |
|-----------|----------|
| 1 | Maximum detail (small prints) |
| 3 | Balanced ⭐ |
| 5 | Smooth (medium prints) |
| 7 | Very smooth (large prints) |
| 9 | Ultra-smooth (rough inputs) |

### Profiling Your System

Run this test sequence to profile your hardware:

```yaml
# Save each config in config.yaml and time it

profile_test_1:  # Baseline
  marigold_steps: 10
  marigold_ensemble: 3
  marigold_processing_res: 768

profile_test_2:  # Ensemble impact
  marigold_steps: 10
  marigold_ensemble: 5
  marigold_processing_res: 768

profile_test_3:  # Resolution impact
  marigold_steps: 10
  marigold_ensemble: 3
  marigold_processing_res: 1024
```

**Record your results:**
- Test 1 time: _____ (baseline)
- Test 2 time: _____ (ensemble +2 = +___% time)
- Test 3 time: _____ (resolution +256 = +___% time)

This helps predict timing for other configurations.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
black .
flake8 .
```

---

## License

This project uses components with different licenses:

- **Pipeline code**: MIT License
- **Marigold model**: Apache 2.0 License
- **Dependencies**: See individual package licenses

See [LICENSE](LICENSE) for full details.

---

## Citation

If you use this pipeline or Marigold in academic work, please cite:

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

For the pipeline itself:
```bibtex
@software{2d_to_3d_pipeline,
  title={2D to 3D Pipeline - Bas-Relief Generator},
  author={Kevin Burt},
  year={2025},
  url={https://github.com/kevdawg/2d-to-3d}
}
```

---

## Acknowledgments

- **Marigold Team** - ETH Zürich for excellent depth estimation
- **Google** - For Gemini and Vertex AI APIs
- **Hugging Face** - For Diffusers library and model hosting
- **Trimesh** - For robust 3D mesh processing
- **Remove.bg** - For high-quality background removal API
- **Real-ESRGAN** - For AI image upscaling

---

## FAQ

### Q: Do I need a GPU?
**A:** No, but highly recommended. CPU processing is 10-20× slower. A GTX 1660 or better makes a huge difference.

### Q: Can I use my own photos?
**A:** Yes! Place them in `data/Photos/` folder. The pipeline will auto-enhance and process them.

### Q: What image formats are supported?
**A:** Input: PNG, JPG, JPEG, BMP. Output: PNG (depth), STL/GLB/OBJ (3D models).

### Q: How do I get better depth maps?
**A:** 1) Use AI enhancement, 2) Remove busy backgrounds, 3) Use high quality preset, 4) Ensure input has clear edges and details, 5) Take photos in good lighting

### Q: Can I edit the 3D models?
**A:** Yes! Use Blender (free), MeshLab, Fusion 360, or any 3D software. OBJ format is most compatible.

### Q: The depth map looks inverted?
**A:** This is normal. Dark = far (background), light = near (foreground). The extrusion handles this correctly.

### Q: Can I use this commercially?
**A:** Check licenses: Marigold is Apache 2.0 (permissive). Google APIs have their own ToS. Pipeline code is MIT (permissive). The maker of this tool would request a voluntary donation so that everything can remain free and open-source.

### Q: How do I make the relief deeper/shallower?
**A:** Adjust `emboss` in config.yaml. Higher = deeper relief. Range: 0.1 (subtle) to 0.8 (dramatic). Likely your model cmes out looking like a pillar at which point you can bring forward the background by setting `far_offset` to a number smaller than 1 to focus the depth on the important parts of the model.

### Q: Why is depth generation so slow?
**A:** Marigold runs AI inference which is compute-intensive. Solutions:
- Use GPU (10-20× faster)
- Lower resolution (768 instead of 1024)
- Reduce ensemble (3 instead of 5)
- Fewer steps (10 instead of 20)

### Q: Can I process videos?
**A:** Not yet. Video support is not planned for a future release. Currently: images only.

### Q: What's the largest image I can process?
**A:** No hard limit, but:
- AI enhancement: ~4096×4096 before auto-fallback to Lanczos
- Depth generation: Limited by GPU VRAM (~2048×2048 on 8GB GPU)
- Large images are auto-resized for processing

---

## Roadmap

### In Progress
- [x] Basic pipeline functionality
- [x] AI image generation (Gemini + Imagen)
- [x] Background removal integration
- [x] AI enhancement with Real-ESRGAN
- [x] Performance timing and profiling
- [ ] Automated testing framework

### Planned Features
- [ ] **Web interface** - Browser-based UI for easier use
- [ ] **Mesh repair** - Automatic mesh cleanup and hole filling
- [ ] **Multi-view depth** - Combine multiple angles for better depth
- [ ] **Docker container** - One-command setup
- [ ] **Cloud processing** - Optional cloud compute for faster processing
- [ ] **Mobile app** - Smartphone camera to 3D model
- [ ] **Custom training** - Fine-tune Marigold on specific subjects

### Future Improvements
- More depth estimation models (DepthAnything, ZoeDepth)
- Additional export formats (GLTF, PLY, X3D)
- Texture mapping improvements
- Normal map generation
- Multi-material support
- Print orientation optimization

---

## Support & Community

- **Documentation**: This README and [CREDENTIALS_SETUP.md](CREDENTIALS_SETUP.md)
- **Issues**: Report bugs via [GitHub Issues](https://github.com/yourusername/2d-to-3d-pipeline/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/yourusername/2d-to-3d-pipeline/discussions)
- **Updates**: Watch the repository for new releases

---

## Version History

### v1.0.0 (Current)
- Initial release
- Marigold depth estimation
- Gemini + Imagen 3 generation
- AI enhancement with Real-ESRGAN
- Background removal (Remove.bg + rembg)
- Quality presets (low/medium/high)
- Batch processing
- Performance profiling

---

**Made with ❤️ for makers, artists, and CNC enthusiasts**

*Transform your 2D designs into stunning 3D relief carvings*