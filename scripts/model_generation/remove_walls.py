#!/usr/bin/env python3
"""
Takes a "pinched" 0-thickness bas-relief mesh and makes it a solid,
watertight block by first removing the flat bottom/degenerate walls
and then rebuilding them cleanly.
"""
import trimesh
import numpy as np
from pathlib import Path

def manual_plane_cut(mesh, plane_origin, plane_normal):
    """
    Cut mesh by plane, keeping only the side the normal points TO.
    (This is the brute-force slice you liked).
    """
    plane_origin = np.array(plane_origin)
    plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
    
    # Calculate signed distance of each vertex from plane
    # Positive = on side normal points TO, negative = opposite side
    vertices = mesh.vertices
    distances = np.dot(vertices - plane_origin, plane_normal)
    
    # Keep vertices on POSITIVE side (inside the boundary)
    vertex_keep = distances >= 0
    
    # Keep faces where ALL vertices are kept
    face_keep = np.all(vertex_keep[mesh.faces], axis=1)
    
    # Create new mesh
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[face_keep],
        process=False
    )
    
    # Remove unreferenced vertices
    new_mesh.remove_unreferenced_vertices()
    
    return new_mesh
    
def remove_frame_walls(input_stl, output_stl, frame_thickness=None):
    """
    Remove the 4 outer frame walls using manual plane cuts,
    then add a new, solid bottom to make it watertight.
    """
    print(f"Slicing frame walls from: {input_stl}")
    
    try:
        mesh = trimesh.load(input_stl)
        print(f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"  [!] Failed to load mesh: {e}")
        return

    # Get mesh bounds
    x_min, y_min, z_min = mesh.bounds[0]
    x_max, y_max, z_max = mesh.bounds[1]
    
    # --- 1. SET THE SLICE OFFSET ---
    offset = 0.0
    if frame_thickness is None:
        x_range = x_max - x_min
        y_range = y_max - y_min
        offset = min(x_range, y_range) * 0.005 # 0.5%
        print(f"Auto-detected frame thickness: {offset:.2f} mm")
    else:
        offset = frame_thickness
        print(f"Using user-provided frame thickness: {offset:.2f} mm")

    # Use a tiny offset for 0.0 to prevent floating point errors
    if offset == 0.0:
        offset = 0.001
        print(f"  INFO: frame_thickness is 0, using tiny offset {offset}mm.")

    # --- 2. SLICE OFF THE 4 WALLS (Your Way) ---
    # We define planes and point the normal INWARD
    cuts = [
        ([x_min + offset, 0, 0], [1, 0, 0], "left wall"),   # Keep RIGHT
        ([x_max - offset, 0, 0], [-1, 0, 0], "right wall"),  # Keep LEFT
        ([0, y_min + offset, 0], [0, 1, 0], "front wall"),  # Keep BACK
        ([0, y_max - offset, 0], [0, -1, 0], "back wall")   # Keep FRONT
    ]
    
    mesh_sliced = mesh
    for plane_origin, plane_normal, wall_name in cuts:
        if mesh_sliced.is_empty: break
        mesh_sliced = manual_plane_cut(mesh_sliced, plane_origin, plane_normal)
        print(f"  Sliced {wall_name}: {len(mesh_sliced.faces)} faces remaining")

    if mesh_sliced.is_empty:
        print("  [!] Slicing failed: resulting mesh is empty.")
        mesh.export(output_stl) # Save original
        return
        
    # --- 3. CREATE A NEW BOTTOM "PLATE" ---
    print("  Creating new bottom plate to patch hole...")
    
    # Get the *new* bounds of the sliced mesh
    s_x_min, s_y_min, s_z_min = mesh_sliced.bounds[0]
    s_x_max, s_y_max, s_z_max = mesh_sliced.bounds[1]
    
    # Define the bounds for a new bottom plate
    # It will be 1mm thick, positioned at the bottom
    # We add a tiny (0.1mm) overlap on X/Y to ensure a good weld
    overlap = 0.1
    bottom_plate_bounds = [
        [s_x_min - overlap, s_y_min - overlap, s_z_min - 1.0], # New bottom (1mm thick)
        [s_x_max + overlap, s_y_max + overlap, s_z_min]        # Old bottom Z
    ]
    
    # Create the bottom plate
    mesh_bottom = trimesh.creation.box(bounds=bottom_plate_bounds)
    print(f"  Created new bottom: {len(mesh_bottom.faces)} faces")

    # --- 4. COMBINE AND CLEAN ---
    # Combine the sliced top/bottom with the new perfect bottom
    final_mesh = trimesh.util.concatenate([mesh_sliced, mesh_bottom])
    
    # Weld all the seams together
    final_mesh.merge_vertices()
    final_mesh.fix_normals()
    final_mesh.remove_unreferenced_vertices()

    print(f"Final: {len(final_mesh.vertices)} vertices, {len(final_mesh.faces)} faces")
    
    if final_mesh.is_watertight:
        print(f"  [OK] Mesh is watertight and capped")
    else:
        print(f"  [!] Mesh is NOT watertight (patching failed)")
            
    # Save
    final_mesh.export(output_stl)
    print(f"Saved: {output_stl}")
    
    return output_stl

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make bas-relief STL solid (add walls/bottom)")
    parser.add_argument("--input", required=True, help="Input STL file (bumpy sheet or pinched)")
    parser.add_argument("--output", required=True, help="Output STL file (solid block)")
    
    # We add this argument so the pipeline script doesn't crash,
    # but this script will ignore it.
    parser.add_argument("--frame-thickness", type=float, help="Ignored (this script makes solids)")
    
    args = parser.parse_args()
    
    remove_frame_walls(args.input, args.output, frame_thickness=None)