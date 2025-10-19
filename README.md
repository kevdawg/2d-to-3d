# 2D to 3D Pipeline

Transform 2D images into high-quality 3D models optimized for bas-relief CNC carving and 3D printing. This pipeline uses AI-powered depth estimation (Marigold) to generate detailed depth maps, then converts them into STL, GLB, and OBJ files.

## Features

- 🎨 **AI Image Generation** - Generate images with Gemini (FREE) or Imagen 3 (high quality)
- 📸 **Photo Enhancement** - Automatic preprocessing for optimal depth map quality
- 🗺️ **Depth Map Generation** - State-of-the-art Marigold depth estimation
- 🎯 **3D Model Creation** - Export to STL, GLB, and OBJ formats
- ⚙️ **Quality Presets** - Low, medium, and high quality options
- 🔄 **Batch Processing** - Process multiple images automatically
- 💾 **Checkpoint System** - Resume interrupted processing

## Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Conda** (recommended for environment management)
- **Windows**, **Linux**, or **macOS**

### 2. Clone Repository

```cmd
git clone https://github.com/yourusername/2d-to-3d-pipeline.git
cd 2d-to-3d-pipeline
```

### 3. Set Up Environments

Create the required conda environments:

```cmd
REM Run the automated installer
install.bat
```

Or manually create environments:

```cmd
REM Image generation environment
conda create -n imagen python=3.10 -y
conda activate imagen
pip install -r environments\requirements_imagen.txt

REM Marigold depth estimation environment
conda create -n marigold python=3.10 -y
conda activate marigold
pip install -r environments\requirements_marigold.txt

REM 3D extrusion environment
conda create -n depth-to-3d python=3.10 -y
conda activate depth-to-3d
pip install -r environments\requirements_depth_to_3d.txt

REM Photo preprocessing environment
conda create -n photo-prep python=3.10 -y
conda activate photo-prep
pip install -r environments\requirements_photo_prep.txt
```

**Note:** The Marigold model (~2GB) will download automatically on first use.

### 4. Set Up Credentials

**For basic usage (FREE):**
```cmd
REM Get API key from: https://aistudio.google.com/app/apikey

REM Set permanently (recommended)
setx GEMINI_API_KEY "your-api-key-here"

REM Then restart your command prompt
```

**For high-quality images (requires billing):**
See [`CREDENTIALS_SETUP.md`](CREDENTIALS_SETUP.md) for detailed instructions on setting up Vertex AI.

### 5. Run the Pipeline

The pipeline imports photo preprocessing modules, so run it in the photo-prep environment:

```cmd
REM Activate photo-prep environment
conda activate photo-prep

REM Run the interactive menu
cd pipeline
python 2D_to_3D_pipeline.py

REM Or use the batch file
cd ..
run_pipeline.bat
```

**Note:** The pipeline uses `conda run` to execute scripts in their specific environments (marigold, depth-to-3d, etc.), but the main pipeline itself should run in photo-prep since it imports those modules.

## Usage

### Interactive Menu

The pipeline provides an interactive menu with the following options:

1. **Generate new image (basic quality)** - Use Gemini API (FREE)
2. **Generate new image (high quality)** - Use Imagen 3 (requires billing)
3. **Enhance photos** - Automatically enhance all photos in 2D_files/
4. **Enhance single photo** - Preview and enhance one photo
5. **Batch process all images** - Convert all 2D images to 3D models
6. **Process single image (low quality)** - Fast processing
7. **Process single image (medium quality)** - Balanced quality/speed
8. **Process single image (high quality)** - Maximum detail
9. **Edit default parameters** - Modify config.yaml settings
10. **Quit**

### Command-Line Usage

For automation or scripting:

```cmd
REM Generate image
python scripts\image_generation\generate_with_gemini.py --prompt "oak leaf" --out data\2D_files\leaf.png

REM Enhance photo
python scripts\photo_preprocessing\photo_preprocess.py --input photo.jpg --output enhanced.png --preset balanced

REM Generate depth map
conda run -n marigold python scripts\depth_generation\marigold_cli.py --input enhanced.png --output depth.png --steps 20 --ensemble 5

REM Create 3D model
conda run -n depth-to-3d python scripts\model_generation\extrude_cli.py --input depth.png --output model.stl --width_mm 100
```

## Project Structure

```
2D-to-3D/
│
├── pipeline/
│   ├── 2D_to_3D_pipeline.py        # Main interactive pipeline
│   ├── config.yaml                  # Configuration settings
│   └── run_pipeline.bat             # Windows launcher
│
├── scripts/
│   ├── image_generation/
│   │   ├── generate_with_gemini.py      # FREE tier image generation
│   │   └── generate_with_imagen3.py     # High-quality generation
│   │
│   ├── photo_preprocessing/
│   │   ├── photo_analyzer.py            # Automatic quality analysis
│   │   └── photo_preprocess.py          # Photo enhancement
│   │
│   ├── depth_generation/
│   │   └── marigold_cli.py              # Depth map generation (auto-downloads model)
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
│   ├── marigold_model/              # Auto-downloaded on first use (git ignored)
│   └── rembg_models/                # Auto-downloaded by rembg (git ignored)
│
├── data/
│   ├── 2D_files/                    # Input images (git ignored)
│   ├── 2D_files_enhanced/           # Preprocessed images (git ignored)
│   └── 3D_files/                    # Output 3D models (git ignored)
│
├── install.bat                      # Automated setup script
├── .gitignore
├── README.md
└── CREDENTIALS_SETUP.md
```

## Configuration

Edit `pipeline/config.yaml` to customize:

- **Quality presets** (low/medium/high)
- **Processing parameters** (steps, ensemble size, resolution)
- **3D model settings** (size, smoothing, emboss depth)
- **Photo enhancement** (denoising, sharpening, contrast)

See comments in `config.yaml` for detailed parameter descriptions.

## Troubleshooting

### "GEMINI_API_KEY not set"
- Set the environment variable and restart your terminal
- See [CREDENTIALS_SETUP.md](CREDENTIALS_SETUP.md)

### "Could not find conda"
- Ensure Conda is installed and added to PATH
- Specify full path in `config.yaml`: `conda_exe: "C:/path/to/conda.exe"`

### "Model not found"
- Run `python download_marigold_model.py` to download the model
- Check that `models/marigold_model/` contains the model files

### "Permission denied" (Vertex AI)
- Enable Vertex AI API: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com
- Set up billing: https://console.cloud.google.com/billing
- See [CREDENTIALS_SETUP.md](CREDENTIALS_SETUP.md) for full setup

### Poor depth map quality
- Use the photo enhancement tool (option 3 or 4 in menu)
- Try the "high_quality" preset
- Ensure input image is well-lit with clear details

## Advanced Usage

### Photo Analysis

Automatically analyze photo quality and get enhancement recommendations:

```bash
python image_preprocessing/photo_analyzer.py --input photo.jpg
```

### Custom Enhancement

Fine-tune enhancement parameters:

```bash
python image_preprocessing/photo_enhancer.py \
  --input photo.jpg \
  --output enhanced.png \
  --preset heavy \
  --denoise 18 \
  --clahe 3.0 \
  --sharpen-percent 200
```

### Checkpoint Resume

If processing is interrupted, resume from checkpoint:

```bash
python depth_generation/marigold_cli.py \
  --input image.png \
  --output depth.png \
  --resume
```

## Performance Tips

- **GPU Recommended** - Marigold runs much faster on NVIDIA GPUs with CUDA
- **Batch Processing** - Process multiple images together for efficiency
- **Quality vs Speed** - Use low/medium presets for testing, high for final output
- **Image Size** - Larger images take longer; 1024x1024 is a good balance
- **Checkpoints** - Enable checkpoints for long-running jobs to allow resume

## Output Files

For each processed image, you'll get:

```
3D_files/
└── oak_leaf/
    ├── oak_leaf.png              # Original image
    ├── oak_leaf_bg_removed.png   # Background removed
    ├── oak_leaf_depth_16bit.png  # 16-bit depth map
    ├── oak_leaf.stl              # STL for 3D printing
    ├── oak_leaf.glb              # GLB with lighting/colors
    └── oak_leaf.obj              # OBJ for editing
```

## API Costs

### Gemini API (FREE Tier)
- **Cost:** $0.00
- **Limits:** 15 requests/minute, 1,500 requests/day
- **Quality:** Good for testing and personal projects

### Vertex AI - Imagen 3 (Paid)
- **Cost:** ~$0.04 per image (1024x1024)
- **Limits:** High (pay as you go)
- **Quality:** Excellent, production-ready
- **Note:** Requires Google Cloud billing account

### Marigold (Offline)
- **Cost:** $0.00 (runs locally)
- **Requirements:** Downloaded model (~2GB)
- **Performance:** Faster with GPU

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project uses components with different licenses:

- **Pipeline code:** MIT License
- **Marigold model:** Apache 2.0 License (see extrude.py header)
- **Dependencies:** See individual package licenses

## Citation

If you use Marigold depth estimation in your work, please cite:

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

## Acknowledgments

- **Marigold Team** - For the excellent depth estimation model
- **Google** - For Gemini and Vertex AI APIs
- **Hugging Face** - For Diffusers library and model hosting
- **Trimesh** - For 3D mesh processing

## Support

- **Documentation:** See `CREDENTIALS_SETUP.md` for credential setup
- **Issues:** Report bugs on GitHub Issues
- **Questions:** Open a discussion on GitHub Discussions

## Roadmap

- [ ] Background removal integration
- [ ] Support for more image generation models
- [ ] Web interface for easier use
- [ ] Docker container for simplified setup
- [ ] Additional 3D export formats
- [ ] Video to 3D animation pipeline

## FAQ

### Q: Do I need a GPU?
A: No, but it's highly recommended. Marigold will run on CPU but be significantly slower (10-20x).

### Q: Can I use my own images instead of generating them?
A: Yes! Just place them in `2D_files/` and use the batch processing or enhancement options.

### Q: What image formats are supported?
A: PNG, JPG, and JPEG for input. Output is always PNG for depth maps.

### Q: How do I get better depth maps?
A: Use the photo enhancement tool (option 3), ensure good lighting, use the high-quality preset, and make sure subjects have clear edges and details.

### Q: Can I use this commercially?
A: Check the licenses of the models and APIs you're using. Marigold is Apache 2.0 (permissive). Google APIs have their own terms of service.

### Q: The depth map looks inverted?
A: This is normal. The extrusion script handles the conversion correctly. Dark = far, light = near.

### Q: How do I edit the 3D models?
A: Use Blender (free), MeshLab, or any 3D modeling software that supports STL/OBJ/GLB files.

---

**Made with ❤️ for makers, artists, and CNC enthusiasts**