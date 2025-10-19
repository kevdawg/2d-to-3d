#!/usr/bin/env python3
"""
Background Removal using remove.bg API
High quality background removal ($0.20 per image after 50 free credits/month)
Includes automatic cropping to remove transparent borders.
"""
import os
import sys
from pathlib import Path
from PIL import Image
import requests
import numpy as np


def crop_transparent_borders(image, margin=10):
    """
    Crop transparent borders from RGBA image, leaving a small margin.
    
    Args:
        image: PIL Image in RGBA mode
        margin: Number of pixels to leave around the subject (default: 10)
    
    Returns:
        Cropped PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Get alpha channel
    alpha = img_array[:, :, 3]
    
    # Find non-transparent pixels
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    
    # Get bounding box
    if not rows.any() or not cols.any():
        # Image is completely transparent - don't crop
        return image
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add margin (but don't exceed image bounds)
    height, width = alpha.shape
    y_min = max(0, y_min - margin)
    y_max = min(height, y_max + margin + 1)
    x_min = max(0, x_min - margin)
    x_max = min(width, x_max + margin + 1)
    
    # Calculate crop percentage
    original_pixels = height * width
    cropped_pixels = (y_max - y_min) * (x_max - x_min)
    saved_percent = ((original_pixels - cropped_pixels) / original_pixels) * 100
    
    # Crop image
    cropped = image.crop((x_min, y_min, x_max, y_max))
    
    print(f"    Cropped: {width}x{height} → {cropped.width}x{cropped.height} ({saved_percent:.1f}% reduction)")
    
    return cropped


def remove_background_removebg(input_path, output_path, api_key=None, crop=True, margin=10):
    """
    Remove background using remove.bg API (paid, high quality).
    
    Args:
        input_path: Path to input image
        output_path: Path to save output PNG
        api_key: remove.bg API key (or set REMOVEBG_API_KEY env var)
        crop: Automatically crop transparent borders (default: True)
        margin: Pixels to leave around subject when cropping (default: 10)
    
    Returns:
        Path to output file
    
    Raises:
        RuntimeError: If API call fails
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get('REMOVEBG_API_KEY')
    
    if not api_key:
        raise RuntimeError(
            "remove.bg API key not found!\n"
            "Set environment variable: setx REMOVEBG_API_KEY \"your-api-key\"\n"
            "Get your API key at: https://www.remove.bg/users/sign_up"
        )
    
    print(f"  Removing background with remove.bg API...")
    
    try:
        # Make API request
        with open(input_path, 'rb') as f:
            response = requests.post(
                'https://api.remove.bg/v1.0/removebg',
                files={'image_file': f},
                data={'size': 'auto'},  # 'auto' = full resolution
                headers={'X-Api-Key': api_key},
                timeout=30
            )
        
        # Check for errors
        if response.status_code == 200:
            # Load result as PIL Image
            from io import BytesIO
            result_img = Image.open(BytesIO(response.content))
            
            # Ensure RGBA
            if result_img.mode != 'RGBA':
                result_img = result_img.convert('RGBA')
            
            # Crop transparent borders if requested
            if crop:
                result_img = crop_transparent_borders(result_img, margin)
            
            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(output_path, 'PNG')
            
            # Get credits info from headers
            credits_charged = response.headers.get('X-Credits-Charged', 'unknown')
            credits_remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
            
            print(f"  ✓ Background removed successfully")
            print(f"    Credits charged: {credits_charged}")
            print(f"    Credits remaining: {credits_remaining}")
            
            return output_path
            
        elif response.status_code == 403:
            raise RuntimeError(
                "API key invalid or expired.\n"
                "Check your API key at: https://www.remove.bg/users/sign_in"
            )
        elif response.status_code == 402:
            raise RuntimeError(
                "Insufficient credits!\n"
                "Buy more credits at: https://www.remove.bg/pricing"
            )
        else:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get('errors', [{}])[0].get('title', 'Unknown error')
            raise RuntimeError(f"API error ({response.status_code}): {error_msg}")
            
    except requests.exceptions.Timeout:
        raise RuntimeError("API request timed out. Check your internet connection.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")


def remove_background_rembg(input_path, output_path, model="isnet-general-use", crop=True, margin=10):
    """
    Remove background using rembg (free, offline).
    
    Args:
        input_path: Path to input image
        output_path: Path to save output PNG
        model: Model to use (u2net, isnet-general-use, u2net_human_seg, silueta)
        crop: Automatically crop transparent borders (default: True)
        margin: Pixels to leave around subject when cropping (default: 10)
    
    Returns:
        Path to output file
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        raise RuntimeError("rembg not installed. Install with: pip install rembg")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"  Removing background with rembg ({model})...")
    
    try:
        # Create session with specified model
        session = new_session(model)
        
        # Remove background
        with Image.open(input_path) as input_img:
            output_img = remove(input_img, session=session)
            
            # Ensure RGBA
            if output_img.mode != 'RGBA':
                output_img = output_img.convert('RGBA')
            
            # Crop transparent borders if requested
            if crop:
                output_img = crop_transparent_borders(output_img, margin)
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_img.save(output_path, 'PNG')
        
        print(f"  ✓ Background removed")
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"rembg failed: {e}")


def remove_background(input_path, output_path, method="removebg", crop=True, margin=10, **kwargs):
    """
    Remove background using specified method and optionally crop transparent borders.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output PNG
        method: "removebg" (paid, high quality) or "rembg" (free)
        crop: Automatically crop transparent borders (default: True)
        margin: Pixels to leave around subject when cropping (default: 10)
        **kwargs: Additional arguments (api_key for removebg, model for rembg)
    
    Returns:
        Path to output file
    """
    if method == "removebg":
        return remove_background_removebg(input_path, output_path, kwargs.get('api_key'), crop, margin)
    elif method == "rembg":
        return remove_background_rembg(input_path, output_path, kwargs.get('model', 'isnet-general-use'), crop, margin)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'removebg' or 'rembg'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--method", choices=["removebg", "rembg"], default="removebg",
                       help="Method to use (default: removebg)")
    parser.add_argument("--model", default="isnet-general-use",
                       help="rembg model to use (only for --method rembg)")
    parser.add_argument("--api-key", help="remove.bg API key (or set REMOVEBG_API_KEY env var)")
    parser.add_argument("--no-crop", action='store_true',
                       help="Don't crop transparent borders (keep full size)")
    parser.add_argument("--margin", type=int, default=10,
                       help="Pixels to leave around subject when cropping (default: 10)")
    
    args = parser.parse_args()
    
    try:
        remove_background(
            args.input, 
            args.output, 
            args.method, 
            crop=not args.no_crop,
            margin=args.margin,
            api_key=args.api_key, 
            model=args.model
        )
        print("\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)