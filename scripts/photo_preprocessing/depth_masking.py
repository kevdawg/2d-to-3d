#!/usr/bin/env python3
"""
Depth Map Masking - Apply alpha transparency to depth maps
Removes background from depth maps using the alpha channel from source images.
"""
import numpy as np
from PIL import Image
from pathlib import Path


def mask_depth_with_alpha(depth_path, alpha_source_path, output_path=None):
    """
    Apply alpha mask from source image to depth map.
    Sets background pixels to 65535 (white = far distance) in depth map.
    
    Args:
        depth_path: Path to depth map (grayscale 16-bit PNG)
        alpha_source_path: Path to image with alpha channel (RGBA PNG)
        output_path: Path to save masked depth (if None, overwrites depth_path)
    
    Returns:
        Path to output file
    """
    depth_path = Path(depth_path)
    alpha_source_path = Path(alpha_source_path)
    
    if output_path is None:
        output_path = depth_path
    else:
        output_path = Path(output_path)
    
    # Load depth map - NumPy 2.x loads as int32, so force uint16
    depth_img = Image.open(depth_path)
    depth_array = np.array(depth_img).astype(np.uint16)  # Force uint16
    
    # Load alpha mask from source
    source_img = Image.open(alpha_source_path)
    
    # Check if source has alpha channel
    if source_img.mode != 'RGBA':
        print(f"  Note: Source has no alpha channel, skipping mask")
        return depth_path
    
    # Extract alpha channel
    alpha = np.array(source_img)[:, :, 3]
    
    # Resize alpha to match depth map size if needed
    if alpha.shape[:2] != depth_array.shape[:2]:
        alpha_img = Image.fromarray(alpha).resize(
            (depth_array.shape[1], depth_array.shape[0]),
            Image.Resampling.LANCZOS
        )
        alpha = np.array(alpha_img)
    
    # Create mask (threshold at 10 to keep semi-transparent pixels)
    mask = alpha > 10
    
    # Apply mask - set background to 65535 (white = far distance)
    masked_depth = depth_array.copy()
    masked_depth[~mask] = 65535  # Changed from 0 to 65535
    
    # Calculate stats
    bg_pixels = np.sum(~mask)
    bg_percent = (bg_pixels / mask.size) * 100
    
    # Save as 16-bit (always, since depth maps are always 16-bit)
    result_img = Image.fromarray(masked_depth.astype(np.uint16))
    result_img.save(output_path, format='PNG', bits=16)
    
    print(f"  Masked depth: {bg_percent:.1f}% background set to far distance")
    return output_path


def batch_mask_depths(depth_dir, alpha_source_dir, output_dir=None):
    """
    Batch process: mask multiple depth maps with their corresponding alpha sources.
    
    Matches files by name (e.g., image_depth.png gets masked by image_nobg.png)
    
    Args:
        depth_dir: Directory containing depth maps
        alpha_source_dir: Directory containing images with alpha channels
        output_dir: Directory to save masked depths (if None, overwrites originals)
    """
    depth_dir = Path(depth_dir)
    alpha_source_dir = Path(alpha_source_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all depth maps
    depth_files = list(depth_dir.glob("*_depth*.png"))
    
    if not depth_files:
        print(f"No depth maps found in {depth_dir}")
        return []
    
    print(f"\nMasking {len(depth_files)} depth map(s)...")
    
    results = []
    for depth_file in depth_files:
        # Try to find corresponding alpha source
        # Try different naming patterns
        base_name = depth_file.stem.replace('_depth_16bit', '').replace('_depth', '')
        
        # Try: base_nobg.png, base_enhanced.png, base.png
        possible_sources = [
            alpha_source_dir / f"{base_name}_nobg.png",
            alpha_source_dir / f"{base_name}_enhanced.png",
            alpha_source_dir / f"{base_name}.png",
        ]
        
        alpha_source = None
        for src in possible_sources:
            if src.exists():
                alpha_source = src
                break
        
        if not alpha_source:
            print(f"  ⚠️ No alpha source found for {depth_file.name}, skipping")
            continue
        
        try:
            output_path = output_dir / depth_file.name if output_dir else None
            result = mask_depth_with_alpha(depth_file, alpha_source, output_path)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Failed to process {depth_file.name}: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mask depth maps with alpha channels")
    parser.add_argument("--depth", required=True, help="Depth map file or directory")
    parser.add_argument("--alpha", required=True, help="Alpha source file or directory")
    parser.add_argument("--output", help="Output file or directory")
    parser.add_argument("--batch", action='store_true', help="Batch process directories")
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            batch_mask_depths(args.depth, args.alpha, args.output)
        else:
            mask_depth_with_alpha(args.depth, args.alpha, args.output)
        
        print("\n✓ Masking complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)