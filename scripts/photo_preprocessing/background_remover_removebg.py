#!/usr/bin/env python3
"""
Background Removal using remove.bg API or rembg.
High quality background removal.
Includes automatic cropping and optional transparent padding.
Supports advanced rembg features:
- Dual-model blending (union/intersection)
- Alpha matting for edge refinement
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
import requests
import numpy as np

OK = "[OK]"
ERR = "[X]"
WARN = "[!]"
TRASH = "[DEL]"
INFO = "[i]"

def add_transparent_padding(image, padding):
    """
    Adds transparent padding around an image.
    
    Args:
        image: PIL Image in RGBA mode
        padding: Pixels to add to all sides
    
    Returns:
        Padded PIL Image
    """
    if padding <= 0:
        return image
    
    print(f"    Adding {padding}px transparent padding...")
    original_size = (image.width, image.height)
    new_size = (image.width + 2 * padding, image.height + 2 * padding)
    
    # Create a new, larger transparent canvas
    padded_img = Image.new('RGBA', new_size, (0, 0, 0, 0)) # Fully transparent
    
    # Paste the original image into the center
    paste_position = (padding, padding)
    padded_img.paste(image, paste_position, image)
    
    print(f"    Padded: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")
    return padded_img


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


def remove_background_removebg(input_path, output_path, api_key=None, crop=True, margin=10, padding=0):
    """
    Remove background using remove.bg API (paid, high quality).
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
            
            # Add padding if requested
            if padding > 0:
                result_img = add_transparent_padding(result_img, padding)
            
            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(output_path, 'PNG')
            
            # Get credits info from headers
            credits_charged = response.headers.get('X-Credits-Charged', 'unknown')
            credits_remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
            
            print(f"  {OK} Background removed successfully")
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


def remove_background_rembg(input_path, output_path, crop=True, margin=10, padding=0, **kwargs):
    """
    Remove background using rembg (free, offline).
    --- UPDATED ---
    Now supports advanced features via **kwargs:
    - model: (str) The primary model to use (e.g., "isnet-general-use")
    - model_secondary: (str) Name of a second model to blend.
    - model_combine_mode: (str) "union" or "intersection".
    - alpha_matting: (bool) Enable/disable edge matting.
    - matting_fg_threshold: (int) Foreground anchor.
    - matting_bg_threshold: (int) Background anchor.
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        raise RuntimeError("rembg not installed. Install with: pip install rembg")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # --- NEW: Get advanced settings from kwargs ---
    # Get the primary model from kwargs
    model = kwargs.get('model', 'isnet-general-use')
    
    model_secondary = kwargs.get('model_secondary', None)
    model_combine_mode = kwargs.get('model_combine_mode', 'none').lower()
    
    use_alpha_matting = kwargs.get('alpha_matting', False)
    matting_fg_threshold = kwargs.get('matting_fg_threshold', 240)
    matting_bg_threshold = kwargs.get('matting_bg_threshold', 10)
    
    # Determine if we need to load a second model
    load_secondary = model_secondary and model_secondary != "none" and model_combine_mode in ["union", "intersection"]

    print(f"  Removing background with rembg ({model})...")
    if load_secondary:
        print(f"  Combining with secondary model: {model_secondary} (mode: {model_combine_mode})")
    if use_alpha_matting and not load_secondary: # Matting only works in single-model mode
        print(f"  Alpha matting enabled (fg: {matting_fg_threshold}, bg: {matting_bg_threshold})")
    # --- END NEW ---
    
    try:
        # Load input image
        with Image.open(input_path) as input_img:
            input_img_exif = ImageOps.exif_transpose(input_img)
        
            # --- NEW: Dual-Model Logic ---
            if load_secondary:
                # 1. Get primary mask
                print(f"    Processing primary model ({model})...")
                session_primary = new_session(model)
                mask_primary = remove(input_img_exif, session=session_primary, only_mask=True)
                
                # 2. Get secondary mask
                print(f"    Processing secondary model ({model_secondary})...")
                
                # --- THIS IS THE FIX ---
                session_secondary = new_session(model_secondary) 
                # --- END OF FIX ---
                
                mask_secondary = remove(input_img_exif, session=session_secondary, only_mask=True)
                
                # 3. Combine masks
                print(f"    Combining masks using '{model_combine_mode}'...")
                mask_primary_arr = np.array(mask_primary)
                mask_secondary_arr = np.array(mask_secondary)
                
                if model_combine_mode == "union":
                    # UNION (OR): Keep pixel if *either* model found it. Fixes "missing ears".
                    combined_mask_arr = np.maximum(mask_primary_arr, mask_secondary_arr)
                else: # intersection
                    # INTERSECTION (AND): Keep pixel only if *both* models found it. Fixes "cow antlers".
                    combined_mask_arr = np.minimum(mask_primary_arr, mask_secondary_arr)
                
                final_mask = Image.fromarray(combined_mask_arr)
                
                # 4. Apply combined mask to original image
                output_img = Image.new("RGBA", input_img_exif.size, (0, 0, 0, 0))
                output_img.paste(input_img_exif, (0, 0), final_mask)

            else:
                # --- Standard Single-Model Path ---
                session = new_session(model)
                output_img = remove(
                    input_img_exif, 
                    session=session,
                    alpha_matting=use_alpha_matting,
                    alpha_matting_foreground_threshold=matting_fg_threshold,
                    alpha_matting_background_threshold=matting_bg_threshold
                )
            # --- END NEW ---

            # Ensure RGBA
            if output_img.mode != 'RGBA':
                output_img = output_img.convert('RGBA')
            
            # Crop transparent borders if requested
            if crop:
                output_img = crop_transparent_borders(output_img, margin)
            
            # Add padding if requested
            if padding > 0:
                output_img = add_transparent_padding(output_img, padding)
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_img.save(output_path, 'PNG')
        
        print(f"  {OK} Background removed")
        return output_path
        
    except Exception as e:
        if "session" not in locals() and "session_primary" not in locals():
             raise RuntimeError(f"rembg failed: Could not load model '{model}'. Check model name and internet connection.")
        raise RuntimeError(f"rembg failed: {e}")


def remove_background(input_path, output_path, method="removebg", crop=True, margin=10, padding=0, **kwargs):
    """
    Remove background using specified method and optionally crop/pad.
    **kwargs are passed to rembg.
    """
    if method == "removebg":
        # Pass api_key from kwargs if it exists
        return remove_background_removebg(
            input_path, output_path, kwargs.get('api_key'), crop, margin, padding=padding
        )
    elif method == "rembg":
        # Pass all kwargs directly to remove_background_rembg
        return remove_background_rembg(
            input_path, 
            output_path, 
            crop, 
            margin, 
            padding=padding,
            **kwargs  # <-- Pass all settings
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'removebg' or 'rembg'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--method", choices=["removebg", "rembg"], default="rembg",
                       help="Method to use (default: rembg)")
    parser.add_argument("--model", default="isnet-general-use",
                       help="rembg model to use (only for --method rembg)")
    parser.add_argument("--api-key", help="remove.bg API key (or set REMOVEBG_API_KEY env var)")
    parser.add_argument("--no-crop", action='store_true',
                       help="Don't crop transparent borders (keep full size)")
    parser.add_argument("--margin", type=int, default=10,
                       help="Pixels to leave around subject when cropping (default: 10)")
    parser.add_argument("--padding", type=int, default=0,
                       help="Add N pixels of transparent padding *after* cropping (default: 0)")

    # Advanced settings
    parser.add_argument("--model-secondary", default=None,
                       help="rembg secondary model to blend (e.g., u2netp)")
    parser.add_argument("--combine-mode", choices=["none", "union", "intersection"], default="none",
                       help="How to blend models (default: none)")
    parser.add_argument("--alpha-matting", action='store_true',
                       help="Enable alpha matting for cleaner edges")
    parser.add_argument("--matting-fg", type=int, default=240,
                       help="Alpha matting foreground threshold (0-255)")
    parser.add_argument("--matting-bg", type=int, default=10,
                       help="Alpha matting background threshold (0-255)")
    
    args = parser.parse_args()
    
    try:
        # Pass arguments to the function
        # We filter out None values for secondary model to avoid issues
        kwargs = {}
        if args.model_secondary: kwargs['model_secondary'] = args.model_secondary
        if args.combine_mode: kwargs['model_combine_mode'] = args.combine_mode
        if args.alpha_matting: kwargs['alpha_matting'] = True
        kwargs['matting_fg_threshold'] = args.matting_fg
        kwargs['matting_bg_threshold'] = args.matting_bg

        remove_background(
            args.input, 
            args.output, 
            args.method, 
            crop=not args.no_crop,
            margin=args.margin,
            padding=args.padding,
            api_key=args.api_key,
            model=args.model,
            **kwargs
        )
        print(f"\n{OK} Success!")
    except Exception as e:
        print(f"\n{ERR} Error: {e}")
        sys.exit(1)