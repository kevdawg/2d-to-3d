#!/usr/bin/env python3
"""
CLI wrapper for mesh repair.
Repairs, optimizes, and decimates a mesh to a target output.

Target can be:
  - Absolute face count (e.g., --target "100000")
  - Density (e.g., --target "1500/cm^2")
"""
import argparse
import sys
from pathlib import Path
import re # <-- Import re

def repair_mesh(input_stl, output_stl, settings):
    """
    Repair and optimize mesh for CNC/3D printing.
    """
    try:
        import pymeshlab
        import trimesh # <-- Import trimesh for bounding box
    except ImportError:
        print("[ERROR] PyMeshLab or Trimesh not installed in this environment")
        print("        Run: pip install pymeshlab trimesh")
        sys.exit(1)
    
    input_stl = Path(input_stl)
    output_stl = Path(output_stl)
    
    print(f"Loading mesh: {input_stl.name}")
    
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(str(input_stl))
    except Exception as e:
        print(f"[ERROR] Failed to load mesh: {e}")
        print("        This can happen if the raw mesh is empty or extremely corrupt.")
        return
    
    original_faces = len(ms.current_mesh().face_matrix())
    original_vertices = len(ms.current_mesh().vertex_matrix())
    
    if original_faces == 0:
        print("[ERROR] Mesh has 0 faces. Cannot repair.")
        return

    print(f"  Original: {original_vertices:,} vertices, {original_faces:,} faces")
    
    # --- NEW: Target Calculation Logic ---
    target_str = settings.get('target', "0")
    target_face_num = 0
    
    # Simplified check for "/cm2" (case-insensitive)
    if "/cm2" in target_str.lower(): # Make lowercase to be safe
        # Density target (e.g., "1500/cm^2")
        try:
            # --- START OF DENSITY FIX ---
            # Use re.match to find the first number
            density_match = re.match(r'^\d+\.?\d*', target_str)
            if not density_match:
                raise ValueError("No number found at start of density string")
                
            density = float(density_match.group(0))
            # --- END OF DENSITY FIX ---
            
            # Use trimesh to get accurate bounds
            mesh_trimesh = trimesh.load(str(input_stl))
            bounds = mesh_trimesh.bounds
            width_mm_actual = bounds[1][0] - bounds[0][0]
            height_mm_actual = bounds[1][1] - bounds[0][1]
            
            area_cm2 = (width_mm_actual / 10.0) * (height_mm_actual / 10.0)
            
            target_face_num = int(area_cm2 * density)
            
            print(f"  Target Density: {density}/cm^2")
            print(f"  Model Area: {area_cm2:.2f} cm^2 ({width_mm_actual:.1f}mm x {height_mm_actual:.1f}mm)")
            print(f"  Calculated Target: {target_face_num:,} faces")
            
        except Exception as e:
            print(f"[WARNING] Could not parse density '{target_str}': {e}")
            target_face_num = 0
    else:
        # Absolute face count target
        try:
            target_face_num = int(target_str)
            print(f"  Target Faces: {target_face_num:,}")
        except ValueError:
            print(f"[WARNING] Invalid target '{target_str}', defaulting to 0")
            target_face_num = 0
    
    # --- END NEW LOGIC ---

    # Step 1: Remove duplicates
    print(f"Removing duplicates...")
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    
    # Step 2: Fix non-manifold geometry
    if settings.get('ensure_manifold', True):
        print(f"Fixing non-manifold edges...")
        ms.meshing_repair_non_manifold_edges()
    
    # Step 3: Fill holes
    if settings.get('fill_holes', True):
        print(f"Filling holes...")
        ms.meshing_close_holes(maxholesize=30)
    
    # Step 4: Re-orient faces
    print(f"Skipping face reorientation (not needed for relief models)")
    
    # Step 5: Smooth (using Taubin to preserve features)
    smooth_iters = settings.get('smooth_iterations', 0)
    if smooth_iters > 0:
        print(f"Smoothing ({smooth_iters} iterations, Taubin method)...")
        ms.apply_coord_taubin_smoothing(
            lambda_=0.5,        # Smoothing amount
            mu=-0.53,           # Shrinkage prevention
            stepsmoothnum=smooth_iters
        )
    
    # Step 6: Decimate
    current_faces = len(ms.current_mesh().face_matrix())
    
    # Only decimate if target is positive and less than current
    if target_face_num > 0 and current_faces > target_face_num:
        # --- START OF CHARMAP FIX ---
        print(f"Decimating: {current_faces:,} -> {target_face_num:,} faces...") # Replaced '→'
        # --- END OF CHARMAP FIX ---
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_face_num,
            preservenormal=True,  # Try to preserve surface normals
            preservetopology=True # Try to prevent topology errors
        )
    elif target_face_num > current_faces:
        print(f"Skipping decimation: Target ({target_face_num:,}) > Current ({current_faces:,})")
    else:
        print(f"Skipping decimation: Target is 0 or invalid.")
    
    # Save
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(output_stl))
    
    final_faces = len(ms.current_mesh().face_matrix())
    final_vertices = len(ms.current_mesh().vertex_matrix())
    
    print(f"  Final: {final_vertices:,} vertices, {final_faces:,} faces")
    
    print(f"[OK] Saved: {output_stl.name}")
    
    return output_stl

def main():
    parser = argparse.ArgumentParser(description="Repair and optimize mesh to a specific target")
    parser.add_argument("--input", required=True, help="Input STL file")
    parser.add_argument("--output", required=True, help="Output STL file")
    
    parser.add_argument("--target", required=True, 
                        help="Target output: integer (100000) or density string ('1500/cm^2')")
    parser.add_argument("--width-mm", required=True, type=float,
                        help="Model width in mm (from config) for density calculation")
    
    parser.add_argument("--smooth", type=int, default=0, help="Smoothing iterations")
    parser.add_argument("--no-fill-holes", action='store_true', help="Skip hole filling")
    parser.add_argument("--no-manifold", action='store_true', help="Skip manifold repair")
    
    args = parser.parse_args()
    
    settings = {
        'target': args.target,
        'width_mm': args.width_mm,
        'smooth_iterations': args.smooth,
        'fill_holes': not args.no_fill_holes,
        'ensure_manifold': not args.no_manifold
    }
    
    try:
        repair_mesh(args.input, args.output, settings)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Mesh repair failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()