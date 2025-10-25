#!/usr/bin/env python3
"""
AI Upscaling for segmented images before depth processing.
Uses Real-ESRGAN or similar for detail enhancement.
"""
from pathlib import Path
from PIL import Image
import subprocess
import sys


def upscale_image_realesrgan(input_path, output_path, scale=4):
    """
    Upscale image using Real-ESRGAN.
    
    Args:
        input_path: Input image
        output_path: Upscaled output
        scale: Upscale factor (2 or 4)
    
    Returns:
        Path to upscaled image
    
    Note: Requires realesrgan-ncnn-vulkan installed
    """
    # Check if Real-ESRGAN is available
    realesrgan_exe = "realesrgan-ncnn-vulkan.exe"  # Windows
    
    if not shutil.which(realesrgan_exe):
        raise RuntimeError(
            "Real-ESRGAN not found. Install from:\n"
            "https://github.com/xinntao/Real-ESRGAN/releases"
        )
    
    cmd = [
        realesrgan_exe,
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", str(scale),
        "-n", "realesrgan-x4plus"  # Model name
    ]
    
    print(f"  Upscaling {scale}x with Real-ESRGAN...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Upscaling failed: {result.stderr}")
    
    return Path(output_path)


def upscale_image_pillow(input_path, output_path, scale=4):
    """
    Fallback: Simple upscaling with Pillow (Lanczos).
    Not AI-based but works without external tools.
    """
    img = Image.open(input_path)
    new_size = (img.width * scale, img.height * scale)
    
    print(f"  Upscaling {scale}x with Pillow (Lanczos)...")
    upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
    upscaled.save(output_path)
    
    return Path(output_path)


def upscale_image(input_path, output_path, scale=4, method="auto"):
    """
    Upscale image for better depth processing.
    
    Args:
        method: "realesrgan", "pillow", or "auto" (try realesrgan, fallback to pillow)
    """
    if method == "auto":
        try:
            return upscale_image_realesrgan(input_path, output_path, scale)
        except RuntimeError:
            print("  Real-ESRGAN not available, using Pillow fallback")
            return upscale_image_pillow(input_path, output_path, scale)
    elif method == "realesrgan":
        return upscale_image_realesrgan(input_path, output_path, scale)
    else:
        return upscale_image_pillow(input_path, output_path, scale)