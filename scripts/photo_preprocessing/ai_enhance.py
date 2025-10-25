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
from PIL import Image, ImageEnhance, ImageFilter
import sys
import subprocess
import importlib
from pathlib import Path
import argparse


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
        print(f"  ✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to install {package_name}: {e}")
        return False


def lazy_import(module_name, package_name=None):
    """
    Try to import a module, install if missing, then retry.
    
    Args:
        module_name: Name to import (e.g., 'cv2')
        package_name: Name to install if different (e.g., 'opencv-python')
    
    Returns:
        Imported module or None if installation failed
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
                print(f"  ✗ Failed to import {module_name} after installation: {e}")
                return None
        return None


def upscale_realesrgan(img, scale=4):
    """
    Upscale using Real-ESRGAN (state-of-the-art AI upscaling).
    Great general purpose - Best for photos, textures, and detailed images.
    
    Performance: Moderate speed on GPU, slow on CPU (~2-4 min for 1024x1024 → 4096x4096)
    Memory: ~4GB RAM for 1024x1024 input
    """
    print(f"Upscaling {scale}x with Real-ESRGAN (AI model)...")
    
    # Lazy import Real-ESRGAN
    realesrgan = lazy_import('realesrgan', 'realesrgan')
    if realesrgan is None:
        print("  ✗ Real-ESRGAN unavailable, falling back to LANCZOS")
        return upscale_lanczos(img, scale)
    
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        # Initialize model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        upsampler = RealESRGANer(
            scale=scale,
            model_path=None,  # Will download automatically
            model=model,
            tile=400,  # Process in tiles to save memory
            tile_pad=10,
            pre_pad=0,
            half=False  # Use FP32 for CPU compatibility
        )
        
        # Convert PIL to numpy array
        img_np = np.array(img)
        
        # Upscale
        output, _ = upsampler.enhance(img_np, outscale=scale)
        
        # Convert back to PIL
        return Image.fromarray(output)
        
    except Exception as e:
        print(f"  ✗ Real-ESRGAN failed: {e}")
        print("  Falling back to LANCZOS...")
        return upscale_lanczos(img, scale)


def upscale_lanczos(img, scale=4):
    """
    Upscale using LANCZOS (Pillow's high-quality resampling).
    Fast! - No AI dependencies, works on any hardware.
    
    Performance: Very fast (~2-5 seconds for any size)
    Memory: Low (~2x input size)
    Quality: Good, but not as detailed as AI methods
    """
    print(f"Upscaling {scale}x with Pillow (Lanczos)...")
    new_size = (img.width * scale, img.height * scale)
    return img.resize(new_size, Image.Resampling.LANCZOS)


def upscale_waifu2x(img, scale=4):
    """
    Upscale using Waifu2x (specialized for anime/illustrations).
    Great for anime characters - Trained on anime/manga, NOT suitable for photos.
    
    Performance: Fast to moderate
    Memory: Moderate
    Use Case: Only use for anime-style images, illustrations, or line art
    """
    print(f"Upscaling {scale}x with Waifu2x (anime model)...")
    
    # Lazy import waifu2x
    # Note: This requires waifu2x-ncnn-vulkan-python
    try:
        waifu2x = lazy_import('waifu2x_ncnn_vulkan_python', 'waifu2x-ncnn-vulkan')
        if waifu2x is None:
            print("  ✗ Waifu2x unavailable, falling back to LANCZOS")
            return upscale_lanczos(img, scale)
        
        from waifu2x_ncnn_vulkan_python import Waifu2x
        
        # Initialize model
        upscaler = Waifu2x(gpuid=0, scale=scale, noise=0)
        
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Upscale
        output = upscaler.process(img_np)
        
        return Image.fromarray(output)
        
    except Exception as e:
        print(f"  ✗ Waifu2x failed: {e}")
        print("  Falling back to LANCZOS...")
        return upscale_lanczos(img, scale)


def upscale_gfpgan(img, scale=4):
    """
    Upscale using GFPGAN (specialized for face restoration).
    Specialized for faces - Best for portraits and close-up faces.
    
    Performance: Moderate on GPU, slow on CPU
    Memory: Moderate to high
    Use Case: ONLY use for images with prominent faces
    Note: Also does face restoration (fixes blur, artifacts)
    """
    print(f"Upscaling {scale}x with GFPGAN (face restoration)...")
    
    # Lazy import GFPGAN
    gfpgan = lazy_import('gfpgan', 'gfpgan')
    if gfpgan is None:
        print("  ✗ GFPGAN unavailable, falling back to LANCZOS")
        return upscale_lanczos(img, scale)
    
    try:
        from gfpgan import GFPGANer
        
        # Initialize model
        restorer = GFPGANer(
            model_path=None,  # Will download automatically
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None  # Don't upscale background separately
        )
        
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Enhance
        _, _, output = restorer.enhance(img_np, has_aligned=False, only_center_face=False, paste_back=True)
        
        return Image.fromarray(output)
        
    except Exception as e:
        print(f"  ✗ GFPGAN failed: {e}")
        print("  Falling back to LANCZOS...")
        return upscale_lanczos(img, scale)


def enhance_clarity(img, strength=1.3):
    """
    Enhance clarity using guided filter (edge-preserving smoothing + sharpening).
    Reduces small noise while enhancing edges.
    
    Args:
        strength: 0.5-2.0 (higher = more aggressive)
    """
    print(f"  Enhancing clarity (strength={strength})...")
    
    # Lazy import opencv
    cv2 = lazy_import('cv2', 'opencv-python')
    if cv2 is None:
        print("  ✗ OpenCV unavailable, skipping clarity enhancement")
        return img
    
    # Convert to numpy
    img_array = np.array(img)
    img_float = img_array.astype(np.float32) / 255.0
    
    # Apply guided filter (edge-preserving smooth)
    try:
        smoothed = cv2.ximgproc.guidedFilter(
            guide=img_float,
            src=img_float,
            radius=4,
            eps=0.01
        )
    except AttributeError:
        # opencv-contrib not installed, use bilateral filter instead
        img_uint8 = img_array.astype(np.uint8)
        smoothed = cv2.bilateralFilter(img_uint8, 5, 50, 50).astype(np.float32) / 255.0
    
    # Enhance by adding back high-frequency details
    details = img_float - smoothed
    enhanced = img_float + details * strength
    
    # Clip and convert back
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)


def enhance_details(img, amount=1.2):
    """
    Enhance fine details using Laplacian pyramid.
    Brings out texture without amplifying noise.
    
    Args:
        amount: 0.5-3.0 (higher = more detail)
    """
    print(f"  Enhancing details (amount={amount})...")
    
    # Lazy import opencv
    cv2 = lazy_import('cv2', 'opencv-python')
    if cv2 is None:
        print("  ✗ OpenCV unavailable, skipping detail enhancement")
        return img
    
    img_array = np.array(img).astype(np.float32)
    
    # Create Gaussian pyramid
    gaussian = [img_array]
    for _ in range(3):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    
    # Create Laplacian pyramid
    laplacian = []
    for i in range(len(gaussian) - 1):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        lap = gaussian[i] - cv2.pyrUp(gaussian[i + 1], dstsize=size)
        laplacian.append(lap * amount)
    
    # Reconstruct with enhanced details
    reconstructed = gaussian[-1]
    for i in range(len(laplacian) - 1, -1, -1):
        size = (laplacian[i].shape[1], laplacian[i].shape[0])
        reconstructed = cv2.pyrUp(reconstructed, dstsize=size) + laplacian[i]
    
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return Image.fromarray(reconstructed)


def sharpen_image(img, radius=2, strength=150):
    """
    Final sharpening pass using unsharp mask.
    
    Args:
        radius: 1-5 (blur radius)
        strength: 50-300 (sharpening percentage)
    """
    print(f"  Sharpening with UnsharpMask (radius={radius}, strength={strength}%)...")
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=strength, threshold=3))


def ai_enhance_image(
    input_path,
    output_path,
    upscale_factor=4,
    upscale_method="realesrgan",
    max_input_size=2048,
    clarity_strength=1.3,
    detail_amount=1.2,
    sharpen_strength=150,
    auto_fallback=True
):
    """
    Complete AI enhancement pipeline.
    
    Args:
        input_path: Path to input image
        output_path: Path to save enhanced image
        upscale_factor: 2, 4, or 8
        upscale_method: "realesrgan", "lanczos", "waifu2x", "gfpgan"
        max_input_size: Max dimension before forcing LANCZOS fallback
        clarity_strength: 0.5-2.0
        detail_amount: 0.5-3.0
        sharpen_strength: 50-300
        auto_fallback: If True, use LANCZOS for oversized images
    
    Returns:
        Path to output file
    """
    print(f"\nAI Enhancement Pipeline")
    print(f"Input: {Path(input_path).name}")
    print(f"Method: {upscale_method} ({upscale_factor}x upscale)\n")
    
    # Load image
    img = Image.open(input_path).convert('RGB')
    original_size = (img.width, img.height)
    print(f"Original size: {img.width}x{img.height}")
    
    # Check if image is too large
    max_dimension = max(img.width, img.height)
    if max_dimension > max_input_size and auto_fallback:
        print(f"\n⚠️  WARNING: Image dimension ({max_dimension}px) exceeds maximum ({max_input_size}px)")
        print(f"⚠️  Large images may cause out-of-memory errors with AI upscaling.")
        print(f"⚠️  Automatically switching to LANCZOS (fast, memory-safe method).")
        print(f"⚠️  To upscale large images with AI: increase 'max_input_size' in config\n")
        upscale_method = "lanczos"
    
    # Step 1: Upscale
    print(f"[1/4] Upscaling {upscale_factor}x...")
    
    upscale_functions = {
        "realesrgan": upscale_realesrgan,
        "lanczos": upscale_lanczos,
        "waifu2x": upscale_waifu2x,
        "gfpgan": upscale_gfpgan
    }
    
    upscale_func = upscale_functions.get(upscale_method.lower(), upscale_realesrgan)
    img = upscale_func(img, upscale_factor)
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
    
    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, 'PNG', quality=100)
    
    print(f"\n✅ Enhancement complete!")
    print(f"   Original: {original_size[0]}x{original_size[1]}")
    print(f"   Enhanced: {img.width}x{img.height}")
    print(f"   Saved to: {output_path}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered image enhancement for depth map optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Upscaling Methods:
  realesrgan  - Great general purpose (recommended for photos)
  lanczos     - Fast! No AI dependencies
  waifu2x     - Great for anime characters (NOT for photos)
  gfpgan      - Specialized for faces (portraits only)

Examples:
  # Upscale with Real-ESRGAN (best quality)
  python ai_enhance.py --input photo.jpg --output enhanced.png --upscale 4 --method realesrgan
  
  # Fast upscale with LANCZOS (no AI)
  python ai_enhance.py --input photo.jpg --output enhanced.png --upscale 4 --method lanczos
  
  # Face restoration with GFPGAN
  python ai_enhance.py --input portrait.jpg --output enhanced.png --upscale 4 --method gfpgan
        """
    )
    
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--upscale", type=int, default=4, choices=[2, 4, 8],
                       help="Upscale factor (default: 4)")
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
            auto_fallback=not args.no_fallback
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()