#!/usr/bin/env python3
"""
AI-Powered Image Enhancement for Depth Map Optimization
Upscales and enhances images using state-of-the-art AI models.

Supports multiple upscaling methods:
- Real-ESRGAN: Great general purpose (recommended for photos)
- LANCZOS: Fast, no AI dependencies (good fallback)
- Waifu2x: Great for anime characters (NOT for photos)
- GFPGAN: Specialized for faces (portraits only)

Dependencies are auto-installed if missing (requires internet connection).
"""

import numpy as np
# --- START OF FIX (Import ImageOps) ---
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
# --- END OF FIX ---
import sys
import subprocess
import importlib
from pathlib import Path
import argparse
import platform
import os
import urllib.request 

OK = "[OK]"
ERR = "[X]"
WARN = "[!]"
TRASH = "[DEL]"
INFO = "[i]"


def install_package_in_current_env(package_name):
    """
    Install package in the currently active Python/conda environment.
    Uses the current Python interpreter to ensure correct environment.
    """
    print(f"  Installing {package_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"  {OK} {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  {ERR} Failed to install {package_name}: {e}")
        return False


def lazy_import(module_name, package_name=None):
    """
    Try to import a module, install if missing, then retry.
    """
    if package_name is None:
        package_name = module_name
    
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"  {module_name} not found, attempting to install...")
        if install_package_in_current_env(package_name):
            try:
                return importlib.import_module(module_name)
            except ImportError as e:
                print(f"  {ERR} Failed to import {module_name} after installation: {e}")
                return None
        return None


def get_model_cache_dir(model_cache_dir=None, sub_dir=""):
    """Helper to get and create the model cache directory."""
    if model_cache_dir is None:
        model_cache_dir = Path(__file__).parent.parent.parent / "models"
    
    cache_path = Path(model_cache_dir) / sub_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def download_model_if_missing(model_path: Path, url: str, model_name: str):
    """Downloads a model file if it doesn't exist."""
    if not model_path.exists():
        print(f"  Downloading {model_name} model...")
        try:
            urllib.request.urlretrieve(url, str(model_path))
            print(f"  {OK} Model cached: {model_path.name}")
        except Exception as e:
            print(f"  {ERR} Failed to download {model_name}: {e}")
            raise RuntimeError(f"Could not download {model_name}")
    else:
        print(f"  Using cached model: {model_path.name}")


def upscale_realesrgan(img, scale=4, model_cache_dir=None):
    """Upscale image using Real-ESRGAN AI model."""
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
        import cv2
        
        print(f"Upscaling {scale}x with Real-ESRGAN (AI model)...")
        
        if scale == 4:
            model_name = 'RealESRGAN_x4plus'
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        else:
            model_name = 'RealESRGAN_x2plus'
            netscale = 2
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            if scale == 1:
                scale = 2
        
        cache_dir = get_model_cache_dir(model_cache_dir, "RealESRGAN")
        model_path = cache_dir / f"{model_name}.pth"
        download_model_if_missing(model_path, model_url, model_name)
        
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(model_path),
            model=model,
            tile=0, tile_pad=10, pre_pad=0, half=False,
            device=device
        )
        
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(output_rgb)
        
    except Exception as e:
        print(f"  {ERR} Real-ESRGAN failed: {e}")
        print(f"  Falling back to LANCZOS...")
        return upscale_lanczos(img, scale)


def upscale_lanczos(img, scale=4):
    """Upscale using LANCZOS (Pillow's high-quality resampling)."""
    if scale == 1:
        print(f"Upscaling 1x with Pillow (Lanczos)...")
        return img  # No upscaling needed
    
    print(f"Upscaling {scale}x with Pillow (Lanczos)...")
    new_size = (img.width * scale, img.height * scale)
    return img.resize(new_size, Image.Resampling.LANCZOS)


def upscale_waifu2x(img, scale=4):
    """Upscale using Waifu2x (specialized for anime/illustrations)."""
    print(f"Upscaling {scale}x with Waifu2x (anime model)...")
    
    try:
        waifu2x = lazy_import('waifu2x_ncnn_vulkan_python', 'waifu2x-ncnn-vulkan')
        if waifu2x is None:
            raise ImportError("Waifu2x not installed")
        
        from waifu2x_ncnn_vulkan_python import Waifu2x
        
        upscaler = Waifu2x(gpuid=0, scale=scale, noise=0)
        img_np = np.array(img)
        output = upscaler.process(img_np)
        
        return Image.fromarray(output)
        
    except Exception as e:
        print(f"  {ERR} Waifu2x failed: {e}")
        print("  Falling back to LANCZOS...")
        return upscale_lanczos(img, scale)


def upscale_gfpgan(img, scale=4, model_cache_dir=None):
    """Upscale using GFPGAN (specialized for face restoration)."""
    print(f"Upscaling {scale}x with GFPGAN (face restoration)...")
    
    gfpgan = lazy_import('gfpgan', 'gfpgan')
    if gfpgan is None:
        print(f"  {ERR} GFPGAN unavailable, falling back to LANCZOS")
        return upscale_lanczos(img, scale)
    
    try:
        from gfpgan import GFPGANer
        
        model_name = 'GFPGANv1.4'
        model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        
        cache_dir = get_model_cache_dir(model_cache_dir, "GFPGAN")
        model_path = cache_dir / f"{model_name}.pth"
        download_model_if_missing(model_path, model_url, model_name)

        restorer = GFPGANer(
            model_path=str(model_path),
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        
        img_np = np.array(img)
        _, _, output = restorer.enhance(img_np, has_aligned=False, only_center_face=False, paste_back=True)
        
        return Image.fromarray(output)
        
    except Exception as e:
        print(f"  {ERR} GFPGAN failed: {e}")
        print("  Falling back to LANCZOS...")
        return upscale_lanczos(img, scale)


def enhance_clarity(img, strength=1.3):
    """Enhance clarity using guided filter (edge-preserving smoothing + sharpening)."""
    print(f"  Enhancing clarity (strength={strength})...")
    
    cv2 = lazy_import('cv2', 'opencv-python')
    if cv2 is None:
        print("  {ERR} OpenCV unavailable, skipping clarity enhancement")
        return img
    
    img_array = np.array(img)
    img_float = img_array.astype(np.float32) / 255.0
    
    try:
        smoothed = cv2.ximgproc.guidedFilter(
            guide=img_float, src=img_float, radius=4, eps=0.01
        )
    except AttributeError:
        img_uint8 = img_array.astype(np.uint8)
        smoothed = cv2.bilateralFilter(img_uint8, 5, 50, 50).astype(np.float32) / 255.0
    
    details = img_float - smoothed
    enhanced = img_float + details * strength
    
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)


def enhance_details(img, amount=1.2):
    """Enhance fine details using Laplacian pyramid."""
    print(f"  Enhancing details (amount={amount})...")
    
    cv2 = lazy_import('cv2', 'opencv-python')
    if cv2 is None:
        print(f"  {ERR} OpenCV unavailable, skipping detail enhancement")
        return img
    
    img_array = np.array(img).astype(np.float32)
    
    gaussian = [img_array]
    for _ in range(3):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    
    laplacian = []
    for i in range(len(gaussian) - 1):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        lap = gaussian[i] - cv2.pyrUp(gaussian[i + 1], dstsize=size)
        laplacian.append(lap * amount)
    
    reconstructed = gaussian[-1]
    for i in range(len(laplacian) - 1, -1, -1):
        size = (laplacian[i].shape[1], laplacian[i].shape[0])
        reconstructed = cv2.pyrUp(reconstructed, dstsize=size) + laplacian[i]
    
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return Image.fromarray(reconstructed)


def sharpen_image(img, radius=2, strength=150):
    """Final sharpening pass using unsharp mask."""
    print(f"  Sharpening with UnsharpMask (radius={radius}, strength={strength}%)...")
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=strength, threshold=3))


def ai_enhance_image(
    input_path,
    output_path,
    upscale_factor=1,
    upscale_method="realesrgan",
    max_input_size=2048,
    clarity_strength=1.3,
    detail_amount=1.2,
    sharpen_strength=150,
    auto_fallback=True, 
    model_cache_dir=None
):

    """
    Complete AI enhancement pipeline.
    """
    print(f"\nAI Enhancement Pipeline")
    print(f"Input: {Path(input_path).name}")
    print(f"Method: {upscale_method} ({upscale_factor}x upscale)\n")
    
    # Load image
    img_with_alpha = Image.open(input_path)
    
    # --- START OF FIX: Apply EXIF orientation ---
    # This reads the metadata, rotates the pixels, and clears the tag.
    img_with_alpha = ImageOps.exif_transpose(img_with_alpha)
    # --- END OF FIX ---

    original_size = (img_with_alpha.width, img_with_alpha.height)
    print(f"Original size: {img_with_alpha.width}x{img_with_alpha.height}")

    if img_with_alpha.mode == 'RGBA':
        print("   Detected RGBA image, preserving alpha channel.")
        alpha_channel = img_with_alpha.getchannel('A')
        img = img_with_alpha.convert('RGB') # Work on RGB channels
        has_alpha = True
    else:
        img = img_with_alpha.convert('RGB')
        alpha_channel = None
        has_alpha = False
    
    # Check if image is too large
    max_dimension = max(img.width, img.height)
    if max_dimension > max_input_size and auto_fallback:
        print(f"\n{WARN}  WARNING: Image dimension ({max_dimension}px) exceeds maximum ({max_input_size}px)")
        print(f"{WARN}  Large images may cause out-of-memory errors with AI upscaling.")
        print(f"{WARN}  Automatically switching to LANCZOS (fast, memory-safe method).")
        print(f"{WARN}  To upscale large images with AI: increase 'max_input_size' in config\n")
        upscale_method = "lanczos"
    
   # Step 1: Upscale
    print(f"[1/4] Upscaling {upscale_factor}x...")

    if upscale_method.lower() == "realesrgan":
        img = upscale_realesrgan(img, upscale_factor, model_cache_dir)
    elif upscale_method.lower() == "lanczos":
        img = upscale_lanczos(img, upscale_factor)
    elif upscale_method.lower() == "waifu2x":
        img = upscale_waifu2x(img, upscale_factor)
    elif upscale_method.lower() == "gfpgan":
        img = upscale_gfpgan(img, upscale_factor, model_cache_dir)
    else:
        img = upscale_realesrgan(img, upscale_factor, model_cache_dir)

    print(f"  Upscaled to: {img.width}x{img.height}")
    
    # Step 2: Clarity Enhancement
    print(f"[2/4] Clarity Enhancement...")
    img = enhance_clarity(img, clarity_strength)
    
    # Step 3: Detail Enhancement
    print(f"[3/4] Detail Enhancement...")
    img = enhance_details(img, detail_amount)
    
    # Step 4: Final Sharpening
    print(f"[4/4] Final Sharpening...")
    img = sharpen_image(img, radius=2, strength=sharpen_strength)
    
    # Re-apply Alpha
    if has_alpha and alpha_channel:
        print("   Re-applying alpha channel...")
        new_size = (img.width, img.height)
        print(f"   Upscaling alpha channel to {new_size}...")
        alpha_upscaled = alpha_channel.resize(new_size, Image.Resampling.LANCZOS)
        img.putalpha(alpha_upscaled)

    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, 'PNG', quality=100)
    
    print(f"\n{OK} Enhancement complete!")
    print(f"   Original: {original_size[0]}x{original_size[1]}")
    print(f"   Enhanced: {img.width}x{img.height}")
    print(f"   Saved to: {output_path}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered image enhancement for depth map optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--upscale", type=int, default=1,
                       choices=[1, 2, 4, 8],
                       help="Upscale factor (default: 1)")
    parser.add_argument("--method", default="realesrgan",
                       choices=["realesrgan", "lanczos", "waifu2x", "gfpgan"],
                       help="Upscaling method (default: realesrgan)")
    parser.add_argument("--max-size", type=int, default=2048,
                       help="Max input dimension before forcing LANCZOS (default: 2048)")
    parser.add_argument("--clarity", type=float, default=1.3,
                       help="Clarity enhancement strength (default: 1.3)")
    parser.add_argument("--detail", type=float, default=1.2,
                       help="Detail enhancement amount (default: 1.2)")
    parser.add_argument("--sharpen", type=int, default=150,
                       help="Sharpening strength (default: 150)")
    parser.add_argument("--no-fallback", action='store_true',
                       help="Disable automatic LANCZOS fallback for large images")
    
    # Get model cache dir from config logic
    try:
        with open(Path(__file__).parent.parent.parent / "pipeline" / "config.yaml", "r") as f:
            import yaml
            cli_cfg = yaml.safe_load(f)
            models_cfg = cli_cfg.get("models", {})
            DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / models_cfg.get("marigold", "models/marigold_model")
            DEFAULT_CACHE_DIR = DEFAULT_CACHE_DIR.parent 
    except Exception:
        DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "models"

    parser.add_argument("--model-cache-dir", default=str(DEFAULT_CACHE_DIR),
                       help="Directory to cache downloaded models")
    
    args = parser.parse_args()
    
    try:
        ai_enhance_image(
            args.input,
            args.output,
            upscale_factor=args.upscale,
            upscale_method=args.method,
            max_input_size=args.max_size,
            clarity_strength=args.clarity,
            detail_amount=args.detail,
            sharpen_strength=args.sharpen,
            auto_fallback=not args.no_fallback,
            model_cache_dir=args.model_cache_dir
        )
    except Exception as e:
        print(f"\n{ERR} Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()