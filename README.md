# 2D to 3D Pipeline

Transform 2D images into high-quality 3D models optimized for bas-relief CNC carving and 3D printing. This pipeline uses AI-powered depth estimation (Marigold) to generate detailed depth maps, then converts them into STL, GLB, and OBJ files.

## Features

- 🎨 **AI Image Generation** - Generate images with Gemini (FREE) or Imagen 3 (high quality)
- 📸 **Photo Enhancement** - Automatic preprocessing for optimal depth map quality
- 🎯 **Regional Processing** - Detect faces/subjects and apply custom detail levels
- 🗺️ **Depth Map Generation** - State-of-the-art Marigold depth estimation
- 🏗️ **3D Model Creation** - Export to STL, GLB, and OBJ formats
- ⚙️ **Quality Presets** - Low, medium, and high quality options
- 🔄 **Batch Processing** - Process multiple images automatically
- 💾 **Checkpoint System** - Resume interrupted processing

## Quick Start

### 1. Prerequisites

**Required Software:**
- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **Conda** - [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- **Git** - [Download](https://git-scm.com/downloads)

**System Requirements:**
- **RAM:** 8GB minimum, 16GB+ recommended
- **Storage:** 10GB free space (for models and processing)
- **GPU:** Optional but highly recommended (NVIDIA with CUDA support)

**Installation Verification:**
```bash
# Verify installations
python --version   # Should show 3.10 or higher
conda --version    # Should show conda version
git --version      # Should show git version
pip --version      # Should show pip version
```

If any are missing, install them from the links above before proceeding.

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/2d-to-3d-pipeline.git
cd 2d-to-3d-pipeline
```

### 3. Set Up Environments

Run the automated installer:

```bash
install.bat  # Windows
# or
./install.sh  # Linux/Mac (if provided)
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

**Note:** Marigold model (~2GB) and SAM model (~2.4GB) download automatically on first use.

### 4. Set Up API Credentials

#### For Basic Usage (FREE - Gemini API)

1. Get your API key: https://aistudio.google.com/app/apikey
2. Set environment variable:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
# Make it permanent:
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your-api-key-here', 'User')
```

**Windows (CMD):**
```cmd
setx GEMINI_API_KEY "your-api-key-here"
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-api-key-here"
# Make it permanent (add to ~/.bashrc or ~/.zshrc):
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
```

3. Restart your terminal

#### For High-Quality Images (Imagen 3 via Vertex AI)

**Requirements:**
- Google Cloud account
- Billing enabled (~$0.04 per image)

**Setup Steps:**

1. **Create Google Cloud Project**
   - Go to: https://console.cloud.google.com/
   - Create new project or select existing one
   - Note your Project ID

2. **Enable Vertex AI API**
   - Go to: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com
   - Click "Enable"

3. **Set Up Billing**
   - Go to: https://console.cloud.google.com/billing
   - Link a billing account to your project

4. **Install Google Cloud SDK**
   - Download: https://cloud.google.com/sdk/docs/install
   - Follow installation instructions for your OS

5. **Authenticate**
   ```bash
   gcloud auth application-default login
   ```
   - Follow browser prompts to sign in
   - This creates credentials at:
     - Windows: `%APPDATA%\gcloud\application_default_credentials.json`
     - Linux/Mac: `~/.config/gcloud/application_default_credentials.json`

6. **Set Project Environment Variable**

   **Windows (PowerShell):**
   ```powershell
   $env:GOOGLE_CLOUD_PROJECT="your-project-id"
   [System.Environment]::SetEnvironmentVariable('GOOGLE_CLOUD_PROJECT', 'your-project-id', 'User')
   ```

   **Windows (CMD):**
   ```cmd
   setx GOOGLE_CLOUD_PROJECT "your-project-id"
   ```

   **Linux/Mac:**
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   echo 'export GOOGLE_CLOUD_PROJECT="your-project-id"' >> ~/.bashrc
   ```

7. **Verify Setup**
   ```bash
   gcloud config get-value project
   # Should show your project ID
   ```

**Troubleshooting Vertex AI:**

- **"Permission denied"**: Enable Vertex AI API and ensure billing is set up
- **"Project not found"**: Verify project ID and that you have access
- **"Quota exceeded"**: Check quotas at https://console.cloud.google.com/iam-admin/quotas
- **"Credentials not found"**: Re-run `gcloud auth application-default login`

### 5. Run the Pipeline

```bash
# Activate photo-prep environment
conda activate photo-prep

# Run the interactive menu
cd pipeline
python 2D_to_3D_pipeline.py

# Or use the batch file (Windows)
cd ..
run_pipeline.bat
```

## Usage

### Interactive Menu

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

### Regional Processing (Advanced)

For photos with people or animals, enable regional processing for better results:

1. Edit `pipeline/config.yaml`:
   ```yaml
   region_processing:
     enabled: true
     detection_mode: 'contour'  # Works for any subject
   ```

2. Run pipeline as normal - faces/subjects get high detail, backgrounds get smoothing

See `REGIONAL_PROCESSING.md` for full documentation.

## Project Structure

```
2D-to-3D/
├── pipeline/
│   ├── 2D_to_3D_pipeline.py        # Main interactive pipeline
│   ├── config.yaml                  # Configuration settings
│   └── run_pipeline.bat             # Windows launcher
├── scripts/
│   ├── image_generation/            # AI image generation
│   ├── photo_preprocessing/         # Photo enhancement
│   ├── depth_generation/            # Depth map generation + ROI detection
│   └── model_generation/            # 3D extrusion
├── environments/                    # Conda environment configs
├── models/                          # Auto-downloaded models (git ignored)
├── data/
│   ├── 2D_files/                    # Input images
│   └── 3D_files/                    # Output 3D models
└── README.md
```

## Configuration

Edit `pipeline/config.yaml` to customize:

- **Quality presets** (low/medium/high)
- **Processing parameters** (steps, ensemble size, resolution)
- **3D model settings** (size, smoothing, emboss depth, max height)
- **Photo enhancement** (denoising, sharpening, contrast)
- **Regional processing** (face/subject detection, per-region settings)

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

### Marigold & SAM (Offline)
- **Cost:** $0.00 (runs locally)
- **Requirements:** Downloaded models (~4.4GB total)
- **Performance:** Faster with GPU

## Performance Tips

- **GPU Recommended** - Marigold runs much faster on NVIDIA GPUs with CUDA
- **Batch Processing** - Process multiple images together for efficiency
- **Quality vs Speed** - Use low/medium presets for testing, high for final output
- **Image Size** - Larger images take longer; 1024x1024 is a good balance
- **Regional Processing** - Adds 1-2 minutes but dramatically improves portrait quality
- **Checkpoints** - Enable for long-running jobs to allow resume

## Troubleshooting

### "GEMINI_API_KEY not set"
- Set the environment variable and restart your terminal
- Verify with: `echo %GEMINI_API_KEY%` (Windows) or `echo $GEMINI_API_KEY` (Linux/Mac)

### "Could not find conda"
- Ensure Conda is installed and added to PATH
- Specify full path in `config.yaml`: `conda_exe: "C:/path/to/conda.exe"`

### "Model not found"
- Models download automatically on first use
- Check internet connection
- Verify `models/` directory has sufficient space (~5GB)

### "Permission denied" (Vertex AI)
- Enable Vertex AI API
- Set up billing
- Verify authentication: `gcloud auth list`

### Poor depth map quality
- Use photo enhancement tool (option 3 or 4 in menu)
- Try high-quality preset
- Enable regional processing for portraits
- Ensure input image is well-lit with clear details

### Out of memory
- Use lower quality preset
- Reduce `processing_res` in config
- Use `detection_mode: 'contour'` instead of 'automatic'
- Close other applications

## Roadmap

### Active Development (Priority Order)

1. **✅ Regional Processing** - Automatic face/subject detection with localized enhancement (COMPLETE!)
   - Detect faces, animals, objects automatically
   - Apply high-detail settings to subjects, smoothing to backgrounds
   - Seamless blending for professional results

2. **🚧 Refined Menu & Workflow** - Simplify UI and improve folder structure
   - Clearer menu options with better descriptions
   - Streamlined folder organization
   - Improved error messages and progress feedback
   - Batch operations with progress tracking

3. **📊 Quality Metrics** - Automated depth map quality assessment
   - Depth consistency scoring
   - Detail preservation metrics
   - Noise detection and reporting
   - Automatic setting recommendations based on quality
   - Side-by-side comparison tools

4. **⚡ Hardware Optimization** - Better CPU/GPU support for all systems
   - Mixed precision inference (FP16/INT8) for faster processing
   - Multi-GPU support for distributed processing
   - Optimized CPU paths for systems without GPU
   - Memory management improvements
   - Model quantization for lower VRAM usage

5. **✅ Automated Testing** - Ensure quality and catch regressions
   - Unit tests for core functions
   - Integration tests for full pipeline
   - Reference image comparisons
   - Performance benchmarks
   - Settings validation and optimization

### Planned Features

**Quality & Features:**
- 🔍 **AI Upscaling** - Pre-process images for better depth detail
- 📏 **Scale Calibration** - Add real-world measurements to photos
- 🎯 **Object-Specific Presets** - Optimized settings for:
  - People (portraits, full body)
  - Animals (pets, wildlife)
  - Architecture (buildings, interiors)
  - Nature (trees, landscapes)
  - Jewelry (small objects, ultra-detail)
  - Text & Patterns (flags, signs, documents, logos)
- 📐 **Camera Intrinsics** - Import camera data for accurate depth
- 🎨 **Surface Texture Mapping** - Preserve material texture (not color) in depth

**Performance & Workflow:**
- ☁️ **Cloud Processing** - Offload computation to cloud GPUs
- 🔗 **Blender Addon** - Direct import with proper scaling
- 🗂️ **Project Management** - Save/load settings per project
- 📝 **Processing History** - Track successful settings
- 🔔 **Notifications** - Alert when long jobs complete

**Advanced Tools:**
- 📱 **Mobile Companion App** - Upload photos from phone
- 🌐 **Web Interface** - Browser-based GUI
- 📦 **Docker Container** - Simplified setup
- 🎨 **Additional AI Models** - More generation options

**Specialized Modes:**
- 📜 **Text Relief Mode** - Specialized for embossing text/logos
- 💍 **Ultra-Detail Mode** - Maximum quality for small objects
- 🏛️ **Architecture Mode** - Optimized for buildings/structures
- 🌳 **Landscape Mode** - Optimized for outdoor scenes

### Research & Exploration

- 🎥 **Video Processing** - Generate depth for video frames
- 🎭 **Style Transfer** - Apply artistic styles to depth
- 🔬 **Custom Model Training** - Fine-tune for specific use cases
- 📊 **Benchmark Suite** - Quality comparison tools

**Have ideas?** Open an issue or discussion to suggest features!

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project uses components with different licenses:

- **Pipeline code:** MIT License
- **Marigold model:** Apache 2.0 License
- **SAM model:** Apache 2.0 License
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
- **META** - For Segment Anything Model (SAM)
- **Google** - For Gemini and Vertex AI APIs
- **Hugging Face** - For Diffusers library and model hosting

## Support

- **Documentation:** See individual feature docs (REGIONAL_PROCESSING.md, etc.)
- **Issues:** Report bugs on GitHub Issues
- **Questions:** Open a discussion on GitHub Discussions

---

**Made with ❤️ for makers, artists, and CNC enthusiasts**