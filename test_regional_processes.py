#!/usr/bin/env python3
"""
Standalone test script for regional depth processing.
Tests face detection, SAM segmentation, and regional Marigold processing.

Usage:
    conda activate marigold
    python test_regional_processing.py --input path/to/photo.jpg
"""
import argparse
from pathlib import Path
import sys
import yaml

# Add script directories to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir / "scripts" / "depth_generation"))

from roi_detector import ROIDetector
from region_processor import RegionProcessor
from depth_blender import DepthBlender
from PIL import Image


def test_regional_processing(image_path, use_sam=True, output_dir=None, remove_bg=True):
    """
    Test complete regional processing pipeline.
    
    Args:
        image_path: Path to input image
        use_sam: Use SAM for precise segmentation (downloads model on first use)
        output_dir: Where to save outputs (defaults to same dir as image)
        remove_bg: Remove background before processing (recommended)
    """
    image_path = Path(image_path)
    
    if output_dir is None:
        output_dir = image_path.parent / f"{image_path.stem}_regional_test"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  Regional Depth Processing Test")
    print("="*60)
    print(f"\nInput: {image_path.name}")
    print(f"Output: {output_dir.name}/")
    print(f"SAM: {'Enabled' if use_sam else 'Disabled (boxes only)'}")
    print(f"Background Removal: {'Enabled' if remove_bg else 'Disabled'}\n")
    
    # STEP 0: Remove background if enabled
    working_image = image_path
    if remove_bg:
        print("[0/3] Removing background...")
        try:
            from rembg import remove
            from PIL import Image as PILImage
            
            bg_removed_path = output_dir / "00_background_removed.png"
            with PILImage.open(image_path) as img:
                output_img = remove(img)
                output_img.save(bg_removed_path, 'PNG')
            
            print(f"   ✓ Background removed and saved")
            working_image = bg_removed_path
            
        except ImportError:
            print(f"   ⚠️ rembg not available, skipping background removal")
            print(f"   Install with: pip install rembg")
        except Exception as e:
            print(f"   ⚠️ Background removal failed: {e}")
            print(f"   Continuing with original image...")
    
    # STEP 1: Detect regions
    print(f"\n[1/3] Detecting subjects...")
    detector = ROIDetector(use_sam=use_sam, detection_mode='contour')  # Use contour (lightweight)
    regions = detector.create_region_masks(str(working_image))
    
    num_faces = len(regions['faces'])
    print(f"   ✓ Detected {num_faces} face(s)")
    
    # Visualize detected regions
    vis_path = output_dir / "01_detected_regions.jpg"
    detector.visualize_regions(working_image, regions, vis_path)
    
    # STEP 2: Show preprocessing examples
    print("\n[2/3] Demonstrating preprocessing differences...")
    
    from photo_preprocess import preprocess_photo
    
    # Create face-enhanced version
    if num_faces > 0:
        face_enhanced = output_dir / "02_face_enhanced.png"
        preprocess_photo(
            str(image_path),
            str(face_enhanced),
            preset='balanced',
            denoise_strength=8,
            clahe_clip=3.0,
            sharpen_percent=200,
            enhance_details=1.5,
            save_intermediate=False
        )
        print(f"   ✓ Face enhancement preview: {face_enhanced.name}")
    
    # Create background-smoothed version
    bg_enhanced = output_dir / "03_background_smoothed.png"
    preprocess_photo(
        str(image_path),
        str(bg_enhanced),
        preset='light',
        denoise_strength=15,
        clahe_clip=1.5,
        sharpen_percent=100,
        use_bilateral=True,
        save_intermediate=False
    )
    print(f"   ✓ Background smoothing preview: {bg_enhanced.name}")
    
    # STEP 3: Create visualization of mask blending
    print("\n[3/3] Creating blend visualization...")
    
    blender = DepthBlender(blend_width=30)
    
    # Combine all masks for visualization
    all_masks = regions['faces'] + [regions['background']]
    
    # Create dummy depth maps (just for visualization)
    import numpy as np
    h, w = all_masks[0].shape
    dummy_depths = [np.random.rand(h, w) * 65535 for _ in all_masks]
    
    blend_vis = output_dir / "04_blend_weights.jpg"
    blender.visualize_blend(dummy_depths, all_masks, blend_vis)
    
    print(f"\n{'='*60}")
    print("  Test Complete!")
    print("="*60)
    print(f"\nGenerated files in: {output_dir.name}/")
    print("  01_detected_regions.jpg    - Face detection results")
    if num_faces > 0:
        print("  02_face_enhanced.png       - High detail preprocessing")
    print("  03_background_smoothed.png - Smooth preprocessing")
    print("  04_blend_weights.jpg       - How regions blend")
    print("\nNext steps:")
    print("  1. Review the preprocessing differences")
    print("  2. Check if face detection worked correctly")
    print("  3. If satisfied, enable in config.yaml:")
    print("     region_processing:")
    print("       enabled: true")
    print("  4. Run main pipeline to generate regional depth maps\n")
    
    # Show statistics
    print("Region Statistics:")
    total_pixels = all_masks[0].size
    for i, mask in enumerate(regions['faces']):
        face_pixels = mask.sum()
        percent = (face_pixels / total_pixels) * 100
        print(f"  Face {i+1}: {percent:.1f}% of image")
    
    bg_pixels = regions['background'].sum()
    bg_percent = (bg_pixels / total_pixels) * 100
    print(f"  Background: {bg_percent:.1f}% of image")


def main():
    parser = argparse.ArgumentParser(
        description="Test regional depth processing (face detection + SAM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with SAM (downloads ~2.4GB model on first use)
  python test_regional_processing.py --input photo.jpg
  
  # Test without SAM (faster, uses boxes only)
  python test_regional_processing.py --input photo.jpg --no-sam
  
  # Specify output directory
  python test_regional_processing.py --input photo.jpg --output test_results/
        """
    )
    
    parser.add_argument("--input", required=True, help="Input image to test")
    parser.add_argument("--no-sam", action='store_true', 
                       help="Skip SAM (use bounding boxes only, faster)")
    parser.add_argument("--no-bg-removal", action='store_true',
                       help="Skip background removal (process full image)")
    parser.add_argument("--output", help="Output directory (optional)")
    
    args = parser.parse_args()
    
    try:
        test_regional_processing(
            args.input,
            use_sam=not args.no_sam,
            output_dir=args.output,
            remove_bg=not args.no_bg_removal
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()