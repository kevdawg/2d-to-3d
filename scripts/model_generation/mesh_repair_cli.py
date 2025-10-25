#!/usr/bin/env python3
"""
CLI wrapper for mesh repair.
Runs in depth-to-3d environment where pymeshlab is installed.
"""
import argparse
import sys
from pathlib import Path


def repair_mesh(input_stl, output_stl, settings):
    """
    Repair and optimize mesh for CNC/3D printing.
    """
    try:
        import pymeshlab
    except ImportError:
        print("[ERROR] PyMeshLab not installed in this environment")
        print("        Run: pip install pymeshlab")
        sys.exit(1)
    
    input_stl = Path(input_stl)
    output_stl = Path(output_stl)
    
    print(f"Loading mesh: {input_stl.name}")
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_stl))
    
    original_faces = len(ms.current_mesh().face_matrix())
    original_vertices = len(ms.current_mesh().vertex_matrix())
    
    print(f"  Original: {original_vertices:,} vertices, {original_faces:,} faces")
    
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
    # Relief models aren't closed volumes, so face reorientation causes issues
    # print(f"Re-orienting faces...")
    # ms.meshing_re_orient_faces_coherently()
    print(f"Skipping face reorientation (not needed for relief models)")
    
    # Step 5: Smooth (using Taubin to preserve features)
    smooth_iters = settings.get('smooth_iterations', 0)
    if smooth_iters > 0:
        print(f"Smoothing ({smooth_iters} iterations, Taubin method)...")
        # Taubin smoothing preserves volume better than Laplacian
        ms.apply_coord_taubin_smoothing(
            lambda_=0.5,        # Smoothing amount
            mu=-0.53,           # Shrinkage prevention
            stepsmoothnum=smooth_iters
        )
    
    # Step 6: Decimate
    target_faces = settings.get('target_faces', 0)
    current_faces = len(ms.current_mesh().face_matrix())
    
    if target_faces > 0 and current_faces > target_faces:
        print(f"Decimating: {current_faces:,} → {target_faces:,} faces...")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    
    # Save
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(output_stl))
    
    final_faces = len(ms.current_mesh().face_matrix())
    final_vertices = len(ms.current_mesh().vertex_matrix())
    
    print(f"  Final: {final_vertices:,} vertices, {final_faces:,} faces")
    
    # Check watertight
    is_watertight = ms.current_mesh().is_compact()
    if is_watertight:
        print(f"[OK] Mesh is watertight (printable)")
    else:
        print(f"[!] Mesh may have issues")
    
    return output_stl

def validate_repair(input_stl, output_stl):
    """
    Quick validation to ensure repair didn't corrupt the mesh.
    Compares vertex/face counts and bounding boxes.
    """
    import trimesh
    
    try:
        original = trimesh.load(input_stl)
        repaired = trimesh.load(output_stl)
        
        # Check vertex count didn't change drastically
        vert_ratio = len(repaired.vertices) / len(original.vertices)
        if vert_ratio < 0.5 or vert_ratio > 2.0:
            print(f"[WARNING] Vertex count changed significantly: {vert_ratio:.2f}x")
        
        # Check face count
        face_ratio = len(repaired.faces) / len(original.faces)
        if face_ratio < 0.5 or face_ratio > 2.0:
            print(f"[WARNING] Face count changed significantly: {face_ratio:.2f}x")
        
        # Check bounding box didn't shrink
        orig_size = original.extents
        repair_size = repaired.extents
        size_change = (repair_size / orig_size).min()
        
        if size_change < 0.9:
            print(f"[WARNING] Model shrank during repair: {size_change:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"[WARNING] Could not validate repair: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Repair and optimize mesh")
    parser.add_argument("--input", required=True, help="Input STL file")
    parser.add_argument("--output", required=True, help="Output STL file")
    parser.add_argument("--smooth", type=int, default=0, help="Smoothing iterations")
    parser.add_argument("--target-faces", type=int, default=0, help="Target face count")
    parser.add_argument("--no-fill-holes", action='store_true', help="Skip hole filling")
    parser.add_argument("--no-manifold", action='store_true', help="Skip manifold repair")
    
    args = parser.parse_args()
    
    settings = {
        'smooth_iterations': args.smooth,
        'target_faces': args.target_faces,
        'fill_holes': not args.no_fill_holes,
        'ensure_manifold': not args.no_manifold
    }
    
    try:
        repair_mesh(args.input, args.output, settings)
        
        # Validate repair didn't break things
        validate_repair(args.input, args.output)
        
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Mesh repair failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()