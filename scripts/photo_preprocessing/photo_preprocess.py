#!/usr/bin/env python3
"""
Photo Enhancement Tool for Depth Map Optimization
Standalone preprocessing pipeline - run BEFORE depth generation pipeline.
Optimized for bas-relief CNC carving and 3D printing applications.
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from pathlib import Path
import argparse
import sys


def denoise_nlm(img_array, strength=10):
    """
    Non-Local Means Denoising - removes noise while preserving edges.
    Best for: Sensor noise, grain, JPEG artifacts
    
    Args:
        strength: 3-20 (higher = more denoising, but can blur details)
    """
    print(f"  Denoising (NLM, strength={strength})...")
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    
    return cv2.fastNlMeansDenoisingColored(
        img_array, None,
        h=strength,
        hColor=strength,
        templateWindowSize=7,
        searchWindowSize=21
    )


def guided_filter_smooth(img_array, radius=4, eps=0.01):
    """
    Guided Filter - edge-preserving smoothing, faster than bilateral.
    Best for: Smoothing while keeping boundaries sharp
    
    Args:
        radius: 2-10 (size of filter)
        eps: 0.001-0.1 (smaller = preserve more edges)
    """
    print(f"  Guided filtering (radius={radius}, eps={eps})...")
    
    # Convert to float32 for processing
    img_float = img_array.astype(np.float32) / 255.0
    
    # Apply guided filter to each channel
    result = np.zeros_like(img_float)
    for i in range(3):
        result[:,:,i] = cv2.ximgproc.guidedFilter(
            guide=img_float[:,:,i],
            src=img_float[:,:,i],
            radius=radius,
            eps=eps
        )
    
    return (result * 255).astype(np.uint8)


def bilateral_filter_smooth(img_array, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral Filter - edge-preserving smoothing, high quality.
    Best for: Photos with both smooth and textured regions
    
    Args:
        d: 5-15 (diameter of neighborhood)
        sigma_color: 50-150 (color similarity threshold)
        sigma_space: 50-150 (spatial distance threshold)
    """
    print(f"  Bilateral filtering (d={d}, sigma={sigma_color})...")
    return cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)


def enhance_detail_laplacian(img_array, amount=1.5):
    """
    Detail Enhancement using Laplacian pyramid.
    Best for: Bringing out fine texture without amplifying noise
    
    Args:
        amount: 0.5-3.0 (strength of detail enhancement)
    """
    print(f"  Enhancing details (amount={amount})...")
    
    # Create Gaussian pyramid
    gaussian = img_array.astype(np.float32)
    gaussian_pyramid = [gaussian]
    for i in range(3):
        gaussian = cv2.pyrDown(gaussian)
        gaussian_pyramid.append(gaussian)
    
    # Create Laplacian pyramid
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        laplacian = gaussian_pyramid[i] - cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian_pyramid.append(laplacian)
    
    # Enhance details in Laplacian pyramid
    enhanced_laplacian = []
    for lap in laplacian_pyramid:
        enhanced_laplacian.append(lap * amount)
    
    # Reconstruct image
    reconstructed = gaussian_pyramid[-1]
    for i in range(len(enhanced_laplacian) - 1, -1, -1):
        size = (enhanced_laplacian[i].shape[1], enhanced_laplacian[i].shape[0])
        reconstructed = cv2.pyrUp(reconstructed, dstsize=size)
        reconstructed += enhanced_laplacian[i]
    
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def hdr_tone_mapping(img_array, gamma=1.0, contrast=1.5):
    """
    HDR-style tone mapping - brings out detail in shadows AND highlights.
    Best for: Outdoor photos, harsh lighting, extreme contrast
    
    Args:
        gamma: 0.5-2.0 (brightness curve, 1.0 = neutral)
        contrast: 1.0-3.0 (local contrast enhancement)
    """
    print(f"  HDR tone mapping (gamma={gamma}, contrast={contrast})...")
    
    # Convert to float
    img_float = img_array.astype(np.float32) / 255.0
    
    # Create tone mapper
    tone_map = cv2.createTonemapMantiuk(gamma=gamma, scale=contrast, saturation=1.0)
    
    # Apply tone mapping
    result = tone_map.process(img_float)
    
    # Convert back
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result


def enhance_contrast_clahe(img_array, clip_limit=2.0, tile_size=8):
    """
    CLAHE - Adaptive histogram equalization for local contrast.
    Best for: Poor lighting, underexposed photos, flat lighting
    
    Args:
        clip_limit: 1.0-4.0 (higher = more contrast)
        tile_size: 4-16 (size of local regions)
    """
    print(f"  CLAHE contrast enhancement (clip={clip_limit})...")
    
    # Convert to LAB
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to luminance
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def sharpen_unsharp_mask(img_array, radius=2, percent=150, threshold=3):
    """
    Unsharp Mask - traditional sharpening technique.
    Best for: Making edges crisp, enhancing definition
    
    Args:
        radius: 1-5 (blur radius for mask)
        percent: 50-300 (sharpening strength %)
        threshold: 0-10 (minimum difference to sharpen)
    """
    print(f"  Sharpening (radius={radius}, percent={percent}%)...")
    
    img_pil = Image.fromarray(img_array)
    sharpened = img_pil.filter(ImageFilter.UnsharpMask(
        radius=radius,
        percent=percent,
        threshold=threshold
    ))
    return np.array(sharpened)


def enhance_saturation(img_array, factor=1.2):
    """
    Saturation boost - helps depth model distinguish features.
    
    Args:
        factor: 0.8-1.5 (saturation multiplier, 1.0 = no change)
    """
    if factor == 1.0:
        return img_array
    
    print(f"  Boosting saturation ({factor}x)...")
    img_pil = Image.fromarray(img_array)
    enhancer = ImageEnhance.Color(img_pil)
    return np.array(enhancer.enhance(factor))


def preprocess_photo(
    input_path,
    output_path,
    preset="balanced",
    # Individual overrides
    denoise_strength=None,
    use_guided_filter=None,
    use_bilateral=None,
    enhance_details=None,
    use_hdr=None,
    clahe_clip=None,
    sharpen_radius=None,
    sharpen_percent=None,
    sharpen_threshold=None,
    saturation=None,
    save_intermediate=False
):
    """
    Complete preprocessing pipeline with configurable presets.
    """
    # Define presets
    PRESETS = {
        "light": {
            "denoise_strength": 5,
            "use_guided_filter": False,
            "use_bilateral": False,
            "enhance_details": 0.0,
            "use_hdr": False,
            "clahe_clip": 1.5,
            "sharpen_radius": 2,
            "sharpen_percent": 120,
            "sharpen_threshold": 3,
            "saturation": 1.1
        },
        "balanced": {
            "denoise_strength": 10,
            "use_guided_filter": True,
            "use_bilateral": False,
            "enhance_details": 1.3,
            "use_hdr": True,
            "clahe_clip": 2.0,
            "sharpen_radius": 2,
            "sharpen_percent": 150,
            "sharpen_threshold": 3,
            "saturation": 1.2
        },
        "heavy": {
            "denoise_strength": 15,
            "use_guided_filter": False,
            "use_bilateral": True,
            "enhance_details": 2.0,
            "use_hdr": True,
            "clahe_clip": 3.0,
            "sharpen_radius": 3,
            "sharpen_percent": 180,
            "sharpen_threshold": 2,
            "saturation": 1.3
        }
    }
    
    # Get preset settings
    if preset not in PRESETS:
        print(f"Unknown preset '{preset}', using 'balanced'")
        preset = "balanced"
    
    settings = PRESETS[preset].copy()
    
    # Apply individual overrides
    if denoise_strength is not None: settings["denoise_strength"] = denoise_strength
    if use_guided_filter is not None: settings["use_guided_filter"] = use_guided_filter
    if use_bilateral is not None: settings["use_bilateral"] = use_bilateral
    if enhance_details is not None: settings["enhance_details"] = enhance_details
    if use_hdr is not None: settings["use_hdr"] = use_hdr
    if clahe_clip is not None: settings["clahe_clip"] = clahe_clip
    if sharpen_radius is not None: settings["sharpen_radius"] = sharpen_radius
    if sharpen_percent is not None: settings["sharpen_percent"] = sharpen_percent
    if sharpen_threshold is not None: settings["sharpen_threshold"] = sharpen_threshold
    if saturation is not None: settings["saturation"] = saturation
    
    print(f"\nPreprocessing: {Path(input_path).name}")
    print(f"Preset: {preset}")
    
    # Load image
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img)
    print(f"Size: {img_array.shape[1]}x{img_array.shape[0]}")
    
    intermediate_dir = None
    if save_intermediate:
        intermediate_dir = Path(output_path).parent / f"{Path(output_path).stem}_steps"
        intermediate_dir.mkdir(exist_ok=True)
        Image.fromarray(img_array).save(intermediate_dir / "00_original.png")
    
    step = 1
    
    # Step 1: Denoise (if needed)
    if settings["denoise_strength"] > 0:
        img_array = denoise_nlm(img_array, settings["denoise_strength"])
        if save_intermediate:
            Image.fromarray(img_array).save(intermediate_dir / f"{step:02d}_denoised.png")
            step += 1
    
    # Step 2: Edge-preserving smoothing (choose one)
    if settings["use_bilateral"]:
        img_array = bilateral_filter_smooth(img_array)
        if save_intermediate:
            Image.fromarray(img_array).save(intermediate_dir / f"{step:02d}_bilateral.png")
            step += 1
    elif settings["use_guided_filter"]:
        try:
            img_array = guided_filter_smooth(img_array)
            if save_intermediate:
                Image.fromarray(img_array).save(intermediate_dir / f"{step:02d}_guided.png")
                step += 1
        except AttributeError:
            print("  Note: Guided filter requires opencv-contrib-python")
    
    # Step 3: HDR tone mapping (if enabled)
    if settings["use_hdr"]:
        img_array = hdr_tone_mapping(img_array, gamma=1.0, contrast=1.5)
        if save_intermediate:
            Image.fromarray(img_array).save(intermediate_dir / f"{step:02d}_hdr.png")
            step += 1
    
    # Step 4: CLAHE contrast
    img_array = enhance_contrast_clahe(img_array, settings["clahe_clip"])
    if save_intermediate:
        Image.fromarray(img_array).save(intermediate_dir / f"{step:02d}_contrast.png")
        step += 1
    
    # Step 5: Detail enhancement (if enabled)
    if settings["enhance_details"] > 0:
        img_array = enhance_detail_laplacian(img_array, settings["enhance_details"])
        if save_intermediate:
            Image.fromarray(img_array).save(intermediate_dir / f"{step:02d}_details.png")
            step += 1
    
    # Step 6: Sharpen
    img_array = sharpen_unsharp_mask(
        img_array,
        settings["sharpen_radius"],
        settings["sharpen_percent"],
        settings["sharpen_threshold"]
    )
    if save_intermediate:
        Image.fromarray(img_array).save(intermediate_dir / f"{step:02d}_sharpened.png")
        step += 1
    
    # Step 7: Saturation
    if settings["saturation"] != 1.0:
        img_array = enhance_saturation(img_array, settings["saturation"])
        if save_intermediate:
            Image.fromarray(img_array).save(intermediate_dir / f"{step:02d}_saturation.png")
            step += 1
    
    # Save final result
    result = Image.fromarray(img_array)
    result.save(output_path, 'PNG', quality=100)
    
    print(f"\n✅ Saved to: {output_path}")
    if save_intermediate:
        print(f"   Intermediate steps saved to: {intermediate_dir}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Photo enhancement tool for optimal depth map generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  light    - Minimal enhancement for already good photos
  balanced - Recommended for most photos (default)
  heavy    - Aggressive enhancement for poor quality photos

Examples:
  # Use balanced preset
  python photo_enhancer.py --input photo.jpg --output enhanced.png
  
  # Use heavy preset with custom denoising
  python photo_enhancer.py --input noisy.jpg --output clean.png --preset heavy --denoise 20
  
  # Save intermediate steps to see each stage
  python photo_enhancer.py --input photo.jpg --output enhanced.png --save-steps
        """
    )
    
    parser.add_argument("--input", required=True, help="Input photo path")
    parser.add_argument("--output", required=True, help="Output path for enhanced image")
    parser.add_argument("--preset", choices=["light", "balanced", "heavy"], default="balanced",
                        help="Enhancement preset (default: balanced)")
    
    # Individual parameter overrides
    parser.add_argument("--denoise", type=int, help="Denoise strength (3-20)")
    parser.add_argument("--guided-filter", action='store_true', help="Use guided filter")
    parser.add_argument("--bilateral", action='store_true', help="Use bilateral filter")
    parser.add_argument("--enhance-details", type=float, help="Detail enhancement (0.5-3.0)")
    parser.add_argument("--hdr", action='store_true', help="Enable HDR tone mapping")
    parser.add_argument("--clahe", type=float, help="CLAHE clip limit (1.0-4.0)")
    parser.add_argument("--sharpen-radius", type=int, help="Sharpen radius (1-5)")
    parser.add_argument("--sharpen-percent", type=int, help="Sharpen strength (50-300)")
    parser.add_argument("--sharpen-threshold", type=int, help="Sharpen threshold (0-10)")
    parser.add_argument("--saturation", type=float, help="Saturation boost (0.8-1.5)")
    
    parser.add_argument("--save-steps", action='store_true', 
                        help="Save intermediate images showing each processing step")
    
    args = parser.parse_args()
    
    try:
        preprocess_photo(
            args.input,
            args.output,
            preset=args.preset,
            denoise_strength=args.denoise,
            use_guided_filter=args.guided_filter if args.guided_filter else None,
            use_bilateral=args.bilateral if args.bilateral else None,
            enhance_details=args.enhance_details,
            use_hdr=args.hdr if args.hdr else None,
            clahe_clip=args.clahe,
            sharpen_radius=args.sharpen_radius,
            sharpen_percent=args.sharpen_percent,
            sharpen_threshold=args.sharpen_threshold,
            saturation=args.saturation,
            save_intermediate=args.save_steps
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()