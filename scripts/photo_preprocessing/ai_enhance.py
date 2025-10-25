#!/usr/bin/env python3
"""
AI Enhancement Pipeline for Face Segments
Combines upscaling, face restoration, and sharpening for maximum detail.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


current_file = Path(__file__).resolve()
current_dir = current_file.parent
scripts_dir = current_dir.parent
depth_gen_dir = scripts_dir / "depth_generation"
sys.path.insert(0, str(depth_gen_dir))


def enhance_with_codeformer(input_path, output_path, fidelity=0.7):
    """
    Enhance face with CodeFormer (face restoration AI).
    
    Args:
        input_path: Input face image
        output_path: Enhanced output
        fidelity: 0-1, higher = more faithful to original (0.5-0.9 recommended)
    
    Returns:
        Path to enhanced image
    
    Requires: CodeFormer installed
    Download from: https://github.com/sczhou/CodeFormer
    """
    codeformer = shutil.which("python") or "python"
    codeformer_script = Path("tools/CodeFormer/inference_codeformer.py")
    
    if not codeformer_script.exists():
        raise RuntimeError(
            "CodeFormer not found. Install from:\n"
            "https://github.com/sczhou/CodeFormer"
        )
    
    cmd = [
        codeformer,
        str(codeformer_script),
        "-i", str(input_path),
        "-o", str(output_path.parent),
        "--fidelity_weight", str(fidelity),
        "--bg_upsampler", "realesrgan",
        "--face_upsample"
    ]
    
    print(f"  Enhancing face with CodeFormer (fidelity={fidelity})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"CodeFormer failed: {result.stderr}")
    
    # CodeFormer saves to specific output structure, move to desired location
    # (Implementation depends on CodeFormer's output structure)
    
    return output_path


def enhance_with_gfpgan(input_path, output_path, version="1.4"):
    """
    Enhance face with GFPGAN (face restoration).
    Lighter weight alternative to CodeFormer.
    
    Requires: gfpgan package
    Install: pip install gfpgan
    """
    try:
        from gfpgan import GFPGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
    except ImportError:
        raise RuntimeError("GFPGAN not installed. Run: pip install gfpgan")
    
    print(f"  Enhancing face with GFPGAN v{version}...")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize restorer
    model_path = f"models/GFPGANv{version}.pth"
    if not Path(model_path).exists():
        print(f"  Downloading GFPGAN model...")
        # Auto-download handled by GFPGANer
    
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,  # Don't upscale, just restore
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )
    
    # Read image
    img = np.array(Image.open(input_path).convert('RGB'))
    
    # Restore
    _, _, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    
    # Save
    Image.fromarray(restored_img).save(output_path)
    
    return output_path


def sharpen_unsharp_mask(input_path, output_path, radius=3, percent=200, threshold=3):
    """
    Apply unsharp mask sharpening.
    This is a fallback if AI tools aren't available.
    """
    print(f"  Sharpening with UnsharpMask (radius={radius}, strength={percent}%)...")
    
    img = Image.open(input_path)
    sharpened = img.filter(ImageFilter.UnsharpMask(
        radius=radius,
        percent=percent,
        threshold=threshold
    ))
    sharpened.save(output_path)
    
    return output_path


def enhance_clarity(input_path, output_path, strength=1.5):
    """
    Enhance local contrast and clarity using high-pass filter technique.
    """
    print(f"  Enhancing clarity (strength={strength})...")
    
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img).astype(np.float32)
    
    # Create blurred version
    blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
    blurred_array = np.array(blurred).astype(np.float32)
    
    # High-pass filter (original - blurred)
    high_pass = img_array - blurred_array
    
    # Add back to original with strength multiplier
    enhanced = img_array + (high_pass * strength)
    
    # Clip to valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # Save
    Image.fromarray(enhanced).save(output_path)
    
    return output_path


def enhance_details_laplacian(input_path, output_path, amount=1.5):
    """
    Detail enhancement using Laplacian sharpening.
    Good for bringing out fine texture.
    """
    print(f"  Enhancing details (amount={amount})...")
    
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    
    from scipy import ndimage
    
    # Create Laplacian kernel
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    
    # Apply to each channel
    enhanced = np.zeros_like(img_array)
    for c in range(3):
        edges = ndimage.convolve(img_array[:, :, c], laplacian_kernel)
        enhanced[:, :, c] = img_array[:, :, c] + amount * edges
    
    # Clip and convert back
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    
    Image.fromarray(enhanced).save(output_path)
    
    return output_path


def full_enhancement_pipeline(
    input_path,
    output_path,
    upscale_factor=4,
    use_face_restoration=True,
    use_clarity=True,
    use_sharpening=True,
    method="auto"
):
    """
    Complete enhancement pipeline for face segments.
    
    Pipeline:
    1. Upscale (Real-ESRGAN or Pillow)
    2. Face restoration (GFPGAN/CodeFormer if available)
    3. Clarity enhancement
    4. Final sharpening
    
    Args:
        input_path: Input cropped face
        output_path: Final enhanced output
        upscale_factor: 2 or 4
        use_face_restoration: Apply AI face enhancement
        use_clarity: Apply clarity/detail enhancement
        use_sharpening: Apply final sharpening
        method: "auto", "gfpgan", "codeformer", "basic"
    """
    print(f"\n{'='*60}")
    print(f"  AI Enhancement Pipeline")
    print(f"{'='*60}")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    work_dir = output_path.parent / "enhancement_temp"
    work_dir.mkdir(exist_ok=True)
    
    current_file = input_path
    step = 1
    
    try:
        # STEP 1: Upscale
        if upscale_factor > 1:
            print(f"\n[{step}/{5}] Upscaling {upscale_factor}x...")
            step_output = work_dir / f"step{step}_upscaled.png"
            
            # Try Real-ESRGAN first
            try:
                from segment_preprocess_cli import upscale_realesrgan
                upscale_realesrgan(current_file, step_output, upscale_factor)
            except:
                print("  Real-ESRGAN not available, using Pillow")
                from segment_preprocess_cli import upscale_pillow
                upscale_pillow(current_file, step_output, upscale_factor)
            
            current_file = step_output
            step += 1
        
        # STEP 2: Face Restoration
        if use_face_restoration:
            print(f"\n[{step}/{5}] AI Face Restoration...")
            step_output = work_dir / f"step{step}_face_restored.png"
            
            if method == "auto":
                # Try GFPGAN first (easier to install)
                try:
                    enhance_with_gfpgan(current_file, step_output)
                except:
                    print("  GFPGAN not available, skipping face restoration")
                    step_output = current_file
            elif method == "gfpgan":
                enhance_with_gfpgan(current_file, step_output)
            elif method == "codeformer":
                enhance_with_codeformer(current_file, step_output)
            else:
                step_output = current_file
            
            current_file = step_output
            step += 1
        
        # STEP 3: Clarity Enhancement
        if use_clarity:
            print(f"\n[{step}/{5}] Clarity Enhancement...")
            step_output = work_dir / f"step{step}_clarity.png"
            enhance_clarity(current_file, step_output, strength=1.3)
            current_file = step_output
            step += 1
        
        # STEP 4: Detail Enhancement
        print(f"\n[{step}/{5}] Detail Enhancement...")
        step_output = work_dir / f"step{step}_details.png"
        enhance_details_laplacian(current_file, step_output, amount=1.2)
        current_file = step_output
        step += 1
        
        # STEP 5: Final Sharpening
        if use_sharpening:
            print(f"\n[{step}/{5}] Final Sharpening...")
            sharpen_unsharp_mask(current_file, output_path, radius=2, percent=150, threshold=3)
        else:
            shutil.copy2(current_file, output_path)
        
        print(f"\n{'='*60}")
        print(f"[OK] Enhancement complete!")
        print(f"     Input:  {Image.open(input_path).size}")
        print(f"     Output: {Image.open(output_path).size}")
        print(f"     Saved to: {output_path}")
        print(f"{'='*60}")
        
        return output_path
        
    finally:
        # Optional: Clean up temp files
        # shutil.rmtree(work_dir)
        pass


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Enhancement Pipeline for Face Segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with all enhancements
  python ai_enhance.py --input face.png --output face_enhanced.png --upscale 4
  
  # Skip face restoration (if not installed)
  python ai_enhance.py --input face.png --output face_enhanced.png --no-face-restore
  
  # Basic enhancement only (no AI)
  python ai_enhance.py --input face.png --output face_enhanced.png --method basic
        """
    )
    
    parser.add_argument("--input", required=True, help="Input face image")
    parser.add_argument("--output", required=True, help="Output enhanced image")
    parser.add_argument("--upscale", type=int, default=4, help="Upscale factor (2 or 4)")
    parser.add_argument("--no-face-restore", action="store_true", help="Skip AI face restoration")
    parser.add_argument("--no-clarity", action="store_true", help="Skip clarity enhancement")
    parser.add_argument("--no-sharpen", action="store_true", help="Skip final sharpening")
    parser.add_argument("--method", choices=["auto", "gfpgan", "codeformer", "basic"], 
                       default="auto", help="Enhancement method")
    
    args = parser.parse_args()
    
    try:
        full_enhancement_pipeline(
            args.input,
            args.output,
            upscale_factor=args.upscale,
            use_face_restoration=not args.no_face_restore,
            use_clarity=not args.no_clarity,
            use_sharpening=not args.no_sharpen,
            method=args.method
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())