#!/usr/bin/env python3
"""
CLI wrapper for extrude.py to convert depth maps to 3D models.
Exposes ALL parameters from extrude.py for full control.
"""
import argparse
import sys
import os
import platform
from pathlib import Path

# Windows-safe symbols
def is_windows_cmd():
    if platform.system() != 'Windows':
        return False
    if 'WT_SESSION' in os.environ:
        return False
    return True

if is_windows_cmd():
    OK = "[OK]"
    ERR = "[X]"
else:
    OK = "✅"
    ERR = "❌"

# Import the extrude function from extrude.py
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

try:
    from extrude import extrude_depth_3d
except ImportError as e:
    print(f"ERROR: Could not import extrude_depth_3d from extrude.py: {e}")
    print(f"Expected location: {script_dir / 'extrude.py'}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a depth map to a 3D model (STL, GLB, OBJ)"
    )
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Path to input depth map (16-bit PNG)")
    parser.add_argument("--output", required=True, help="Path for output STL file")
    
    # Basic model parameters
    parser.add_argument("--width_mm", type=float, default=100.0,
                        help="Width of the 3D model in mm (default: 100)")
    parser.add_argument("--max_height_mm", type=float, default=None,
                        help="Maximum height of relief in mm (overrides emboss if set)")
    parser.add_argument("--smoothing", type=int, default=3,
                        help="Median filter size for smoothing, must be odd (default: 3)")
    
    # Depth mapping parameters
    parser.add_argument("--near_offset", type=float, default=0.0,
                        help="Near clipping offset 0.0-1.0 (default: 0.0)")
    parser.add_argument("--far_offset", type=float, default=1.0,
                        help="Far clipping offset 0.0-1.0 (default: 1.0)")
    
    # Frame/border parameters
    parser.add_argument("--emboss", type=float, default=0.3,
                        help="Emboss depth factor 0.1-1.0 (default: 0.3, ignored if max_height_mm set)")
    parser.add_argument("--f_thic", type=float, default=0.05,
                        help="Frame thickness 0.01-0.2 (default: 0.05, use 0 for no border)")
    parser.add_argument("--f_near", type=float, default=-0.15,
                        help="Front frame position -0.5-0.0 (default: -0.15)")
    parser.add_argument("--f_back", type=float, default=0.01,
                        help="Back frame thickness 0.01-0.1 (default: 0.01)")
    
    # Output options
    parser.add_argument("--vertex_colors", type=lambda x: x.lower() == 'true', default=True,
                        help="Include vertex colors from RGB (default: true)")
    parser.add_argument("--scene_lights", type=lambda x: x.lower() == 'true', default=True,
                        help="Add directional lights to GLB (default: true)")
    parser.add_argument("--prepare_for_3d_printing", type=lambda x: x.lower() == 'true', default=False,
                        help="Rotate model for 3D printing orientation (default: false)")
    parser.add_argument("--zip_outputs", type=lambda x: x.lower() == 'true', default=False,
                        help="Compress output files into zip (default: false)")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)
    
    # VALIDATE AND FIX DEPTH MAP FORMAT
    print(f"Validating depth map format...")
    from PIL import Image
    import numpy as np
    
    depth_img = Image.open(input_path)
    print(f"  Input: {depth_img.mode}, {depth_img.size}")
    
    needs_conversion = False
    
    # Check if it's grayscale
    if depth_img.mode not in ['L', 'I', 'I;16']:
        print(f"  Converting from {depth_img.mode} to grayscale...")
        needs_conversion = True
        
        # If RGB/RGBA, use first channel or convert
        if depth_img.mode in ['RGB', 'RGBA']:
            # Use luminosity conversion
            depth_img = depth_img.convert('L')
        else:
            depth_img = depth_img.convert('L')
    
    # Check bit depth and convert to 16-bit if needed
    depth_array = np.array(depth_img)
    
    # Handle multi-channel (shouldn't happen after conversion, but be safe)
    if len(depth_array.shape) > 2:
        print(f"  Extracting first channel from multi-channel depth...")
        depth_array = depth_array[:, :, 0]
        needs_conversion = True
    
    # Ensure 16-bit depth values
    if depth_array.dtype != np.uint16:
        print(f"  Converting to 16-bit depth...")
        if depth_array.max() <= 255:
            # 8-bit, scale to 16-bit
            depth_array = (depth_array.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
        else:
            depth_array = depth_array.astype(np.uint16)
        needs_conversion = True
    
    # If we made changes, save corrected version temporarily
    if needs_conversion:
        import tempfile
        temp_depth = tempfile.NamedTemporaryFile(suffix='_depth_corrected.png', delete=False)
        temp_depth_path = temp_depth.name
        temp_depth.close()
        
        corrected_img = Image.fromarray(depth_array.astype(np.uint16))
        corrected_img.save(temp_depth_path, format='PNG')
        print(f"  Corrected depth map saved temporarily")
        
        # Use corrected path for extrusion
        actual_input_path = temp_depth_path
    else:
        print(f"  Depth map format is valid")
        actual_input_path = str(input_path)
    
    output_path = Path(args.output)
    output_base = str(output_path.with_suffix(''))

    # Convert smoothing to odd integer
    filter_size = int(args.smoothing)
    if filter_size % 2 == 0:
        filter_size += 1  # Make it odd
        print(f"Note: Smoothing adjusted to {filter_size} (must be odd)")
    
    print(f"Converting depth map to 3D model...")
    if args.max_height_mm:
        print(f"Scale: {args.width_mm}mm wide × {args.max_height_mm}mm tall | Smoothing: {filter_size}")
    else:
        print(f"Scale: {args.width_mm}mm | Smoothing: {filter_size} | Emboss: {args.emboss}")
    
    try:
        # Call the extrude function with ALL parameters
        path_glb, path_stl, path_obj = extrude_depth_3d(
            path_depth=str(actual_input_path),
            path_rgb=None,
            path_out_base=output_base,
            output_model_scale=args.width_mm,
            max_height_mm=args.max_height_mm,
            filter_size=filter_size,
            coef_near=args.near_offset,
            coef_far=args.far_offset,
            emboss=args.emboss,
            f_thic=args.f_thic,
            f_near=args.f_near,
            f_back=args.f_back,
            vertex_colors=args.vertex_colors,
            scene_lights=args.scene_lights,
            prepare_for_3d_printing=args.prepare_for_3d_printing,
            zip_outputs=args.zip_outputs,
        )
        
        print(f"{OK} 3D models created successfully!")
        print(f"   STL: {Path(path_stl).name}")
        print(f"   GLB: {Path(path_glb).name}")
        print(f"   OBJ: {Path(path_obj).name}")
        
    except Exception as e:
        print(f"\n{ERR} ERROR: Failed to create 3D model: {e}")
        import traceback
        traceback.print_exc()

        if needs_conversion and 'actual_input_path' in locals() and os.path.exists(actual_input_path):
            os.unlink(actual_input_path)
            
        sys.exit(1)


if __name__ == "__main__":
    main()