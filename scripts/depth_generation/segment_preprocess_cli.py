#!/usr/bin/env python3
"""
CLI for preprocessing cropped segments before Marigold.
Adds context padding and optionally upscales.
"""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageFilter
import shutil


def add_context_padding(cropped_img, padding_percent=10, blur_radius=20):
    """Add blurred padding around segment."""
    w, h = cropped_img.size
    
    pad_w = int(w * padding_percent / 100)
    pad_h = int(h * padding_percent / 100)
    
    new_w = w + 2 * pad_w
    new_h = h + 2 * pad_h
    
    canvas = Image.new('RGB', (new_w, new_h), color=(128, 128, 128))
    canvas.paste(cropped_img, (pad_w, pad_h))
    
    blurred = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
    blurred.paste(cropped_img, (pad_w, pad_h))
    
    return blurred


def upscale_realesrgan(input_path, output_path, scale=4):
    """Upscale with Real-ESRGAN."""
    import subprocess
    
    realesrgan = shutil.which("realesrgan-ncnn-vulkan") or shutil.which("realesrgan-ncnn-vulkan.exe")
    
    if not realesrgan:
        raise RuntimeError("Real-ESRGAN not found in PATH")
    
    cmd = [
        realesrgan,
        "-i", str(input_path),
        "-o", str(output_path),
        "-s", str(scale),
        "-n", "realesrgan-x4plus"
    ]
    
    print(f"Upscaling {scale}x with Real-ESRGAN...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Real-ESRGAN failed: {result.stderr}")
    
    return Path(output_path)


def upscale_pillow(input_path, output_path, scale=4):
    """Upscale with Pillow Lanczos."""
    img = Image.open(input_path)
    new_size = (img.width * scale, img.height * scale)
    
    print(f"Upscaling {scale}x with Pillow (Lanczos)...")
    upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
    upscaled.save(output_path)
    
    return Path(output_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess segment for Marigold")
    parser.add_argument("--input", required=True, help="Input cropped segment")
    parser.add_argument("--output", required=True, help="Output preprocessed image")
    parser.add_argument("--padding", type=int, default=10, help="Context padding percent (default: 10)")
    parser.add_argument("--upscale", type=int, default=0, help="Upscale factor (0=none, 2, 4)")
    parser.add_argument("--upscale-method", choices=["realesrgan", "pillow", "auto"], default="auto")
    parser.add_argument("--no-padding", action="store_true", help="Skip padding step")
    
    args = parser.parse_args()
    
    try:
        img = Image.open(args.input).convert('RGB')
        print(f"Input: {img.size}")
        
        # Step 1: Add padding
        if not args.no_padding:
            img = add_context_padding(img, args.padding)
            print(f"After padding: {img.size}")
        
        # Step 2: Upscale
        if args.upscale > 1:
            temp_path = Path(args.output).parent / "temp_before_upscale.png"
            img.save(temp_path)
            
            if args.upscale_method == "auto":
                try:
                    upscale_realesrgan(temp_path, args.output, args.upscale)
                except RuntimeError:
                    print("Real-ESRGAN not available, using Pillow")
                    upscale_pillow(temp_path, args.output, args.upscale)
            elif args.upscale_method == "realesrgan":
                upscale_realesrgan(temp_path, args.output, args.upscale)
            else:
                upscale_pillow(temp_path, args.output, args.upscale)
            
            temp_path.unlink()
        else:
            img.save(args.output)
        
        final = Image.open(args.output)
        print(f"[OK] Output: {final.size}")
        print(f"     Saved to: {args.output}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()