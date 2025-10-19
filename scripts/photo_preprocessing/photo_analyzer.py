#!/usr/bin/env python3
"""
Photo Analyzer - Automatically recommend enhancement settings for depth map optimization.
Analyzes photo quality metrics and suggests optimal preprocessing parameters.
"""
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import argparse


def analyze_noise_level(img_array):
    """
    Estimate noise level using Laplacian variance in flat regions.
    Returns: noise_score (0-100, higher = more noisy)
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Compute Laplacian variance (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize to 0-100 scale
    noise_score = min(100, variance / 10)
    
    return noise_score


def analyze_contrast(img_array):
    """
    Measure contrast using histogram spread.
    Returns: contrast_score (0-100, lower = needs more contrast)
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Measure spread (standard deviation)
    mean = np.sum(np.arange(256) * hist.flatten()) / np.sum(hist)
    variance = np.sum(((np.arange(256) - mean) ** 2) * hist.flatten()) / np.sum(hist)
    std_dev = np.sqrt(variance)
    
    # Normalize (good contrast has std ~60-80)
    contrast_score = min(100, (std_dev / 80) * 100)
    
    return contrast_score


def analyze_sharpness(img_array):
    """
    Measure image sharpness using gradient magnitude.
    Returns: sharpness_score (0-100, lower = needs more sharpening)
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_magnitude = np.mean(magnitude)
    
    # Normalize (sharp images have mean ~30-50)
    sharpness_score = min(100, (mean_magnitude / 50) * 100)
    
    return sharpness_score


def analyze_dynamic_range(img_array):
    """
    Check for clipped shadows and highlights.
    Returns: (clipped_shadows_%, clipped_highlights_%)
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Count pixels near extremes
    total_pixels = gray.size
    clipped_shadows = np.sum(gray < 10) / total_pixels * 100
    clipped_highlights = np.sum(gray > 245) / total_pixels * 100
    
    return clipped_shadows, clipped_highlights


def analyze_saturation(img_array):
    """
    Measure color saturation.
    Returns: saturation_score (0-100, lower = more desaturated)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    # Mean saturation
    mean_sat = np.mean(saturation)
    
    # Normalize (good saturation is around 100-150)
    saturation_score = min(100, (mean_sat / 150) * 100)
    
    return saturation_score


def recommend_settings(
    noise_score,
    contrast_score,
    sharpness_score,
    clipped_shadows,
    clipped_highlights,
    saturation_score
):
    """
    Based on analysis, recommend enhancement settings.
    """
    recommendations = {
        "preset": "balanced",
        "denoise_strength": 10,
        "use_hdr": False,
        "clahe_clip": 2.0,
        "sharpen_percent": 150,
        "saturation": 1.2,
        "use_detail_enhancement": True,
        "detail_amount": 1.3
    }
    
    reasons = []
    
    # Noise analysis
    if noise_score > 60:
        recommendations["preset"] = "heavy"
        recommendations["denoise_strength"] = 18
        reasons.append(f"High noise detected ({noise_score:.0f}/100) → Heavy denoising")
    elif noise_score > 30:
        recommendations["denoise_strength"] = 12
        reasons.append(f"Moderate noise ({noise_score:.0f}/100) → Standard denoising")
    elif noise_score < 15:
        recommendations["denoise_strength"] = 5
        reasons.append(f"Clean image ({noise_score:.0f}/100) → Light denoising")
    
    # Contrast analysis
    if contrast_score < 50:
        recommendations["clahe_clip"] = 3.0
        reasons.append(f"Low contrast ({contrast_score:.0f}/100) → Strong CLAHE")
    elif contrast_score < 70:
        recommendations["clahe_clip"] = 2.0
        reasons.append(f"Moderate contrast ({contrast_score:.0f}/100) → Standard CLAHE")
    else:
        recommendations["clahe_clip"] = 1.5
        reasons.append(f"Good contrast ({contrast_score:.0f}/100) → Light CLAHE")
    
    # Sharpness analysis
    if sharpness_score < 40:
        recommendations["sharpen_percent"] = 200
        reasons.append(f"Soft image ({sharpness_score:.0f}/100) → Strong sharpening")
    elif sharpness_score < 70:
        recommendations["sharpen_percent"] = 150
        reasons.append(f"Normal sharpness ({sharpness_score:.0f}/100) → Standard sharpening")
    else:
        recommendations["sharpen_percent"] = 100
        reasons.append(f"Already sharp ({sharpness_score:.0f}/100) → Light sharpening")
    
    # Dynamic range analysis
    if clipped_shadows > 5 or clipped_highlights > 5:
        recommendations["use_hdr"] = True
        reasons.append(f"Clipped pixels (shadows:{clipped_shadows:.1f}%, highlights:{clipped_highlights:.1f}%) → Enable HDR")
    
    # Saturation analysis
    if saturation_score < 40:
        recommendations["saturation"] = 1.4
        reasons.append(f"Desaturated ({saturation_score:.0f}/100) → Boost saturation")
    elif saturation_score < 70:
        recommendations["saturation"] = 1.2
        reasons.append(f"Normal saturation ({saturation_score:.0f}/100) → Mild boost")
    else:
        recommendations["saturation"] = 1.0
        reasons.append(f"Good saturation ({saturation_score:.0f}/100) → No boost needed")
    
    # Detail enhancement
    if sharpness_score < 50 and noise_score < 40:
        recommendations["use_detail_enhancement"] = True
        recommendations["detail_amount"] = 2.0
        reasons.append("Low detail + low noise → Strong detail enhancement")
    elif sharpness_score < 70:
        recommendations["detail_amount"] = 1.3
    else:
        recommendations["use_detail_enhancement"] = False
        reasons.append("Already detailed → Skip detail enhancement")
    
    return recommendations, reasons


def analyze_photo(input_path, verbose=True):
    """
    Analyze photo and recommend enhancement settings.
    Returns: (settings_dict, reasons_list, command_string)
    """
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img)
    
    if verbose:
        print(f"\nAnalyzing: {Path(input_path).name}")
        print(f"Size: {img_array.shape[1]}x{img_array.shape[0]}\n")
    
    # Run analyses
    noise_score = analyze_noise_level(img_array)
    contrast_score = analyze_contrast(img_array)
    sharpness_score = analyze_sharpness(img_array)
    clipped_shadows, clipped_highlights = analyze_dynamic_range(img_array)
    saturation_score = analyze_saturation(img_array)
    
    if verbose:
        print("Image Quality Metrics:")
        print(f"  Noise level:      {noise_score:5.1f}/100  {'(high - needs denoising)' if noise_score > 50 else '(acceptable)'}")
        print(f"  Contrast:         {contrast_score:5.1f}/100  {'(low - needs enhancement)' if contrast_score < 60 else '(good)'}")
        print(f"  Sharpness:        {sharpness_score:5.1f}/100  {'(soft - needs sharpening)' if sharpness_score < 60 else '(sharp)'}")
        print(f"  Clipped shadows:  {clipped_shadows:5.1f}%    {'(significant clipping)' if clipped_shadows > 5 else '(minimal)'}")
        print(f"  Clipped highlights:{clipped_highlights:5.1f}%    {'(significant clipping)' if clipped_highlights > 5 else '(minimal)'}")
        print(f"  Saturation:       {saturation_score:5.1f}/100  {'(low - needs boost)' if saturation_score < 60 else '(good)'}")
    
    # Get recommendations
    settings, reasons = recommend_settings(
        noise_score,
        contrast_score,
        sharpness_score,
        clipped_shadows,
        clipped_highlights,
        saturation_score
    )
    
    # Build command string
    input_name = Path(input_path).name
    cmd = f"python photo_enhancer.py --input {input_name} --output enhanced_{input_name}"
    cmd += f" --preset {settings['preset']}"
    
    if settings['denoise_strength'] != 10:
        cmd += f" --denoise {settings['denoise_strength']}"
    if settings['use_hdr']:
        cmd += " --hdr"
    if settings['clahe_clip'] != 2.0:
        cmd += f" --clahe {settings['clahe_clip']}"
    if settings['sharpen_percent'] != 150:
        cmd += f" --sharpen-percent {settings['sharpen_percent']}"
    if settings['saturation'] != 1.2:
        cmd += f" --saturation {settings['saturation']}"
    if settings['use_detail_enhancement']:
        cmd += f" --enhance-details {settings['detail_amount']}"
    
    if verbose:
        print("\nRecommended Settings:")
        print(f"  Preset:           {settings['preset']}")
        print(f"  Denoise strength: {settings['denoise_strength']}")
        print(f"  CLAHE clip limit: {settings['clahe_clip']}")
        print(f"  Sharpen percent:  {settings['sharpen_percent']}")
        print(f"  Saturation boost: {settings['saturation']}")
        print(f"  HDR tone mapping: {'Yes' if settings['use_hdr'] else 'No'}")
        print(f"  Detail enhance:   {'Yes (' + str(settings['detail_amount']) + ')' if settings['use_detail_enhancement'] else 'No'}")
        
        print("\nReasoning:")
        for reason in reasons:
            print(f"  • {reason}")
        
        print("\nTo manually adjust, run:")
        print(f"\n  {cmd}\n")
    
    return settings, reasons, cmd


def main():
    parser = argparse.ArgumentParser(
        description="Analyze photos and recommend optimal enhancement settings for depth mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool analyzes your photo and recommends the best enhancement settings
for generating high-quality depth maps and 3D models.

Example:
  python photo_analyzer.py --input my_photo.jpg
  
  # Then use the suggested command to enhance the photo
  python photo_enhancer.py --input my_photo.jpg --output enhanced.png --preset heavy --denoise 18
        """
    )
    
    parser.add_argument("--input", required=True, help="Input photo to analyze")
    parser.add_argument("--quiet", action='store_true', help="Minimal output (just show command)")
    
    args = parser.parse_args()
    
    try:
        analyze_photo(args.input, verbose=not args.quiet)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())