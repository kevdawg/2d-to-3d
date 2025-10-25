#!/usr/bin/env python3
"""
Trim border frame from extruded STL files.
Removes the outer rectangular frame, keeping only the relief surface.
"""
import numpy as np
import trimesh
from pathlib import Path


def trim_border_frame(input_stl, output_stl, tolerance=0.01):
    """
    Remove border frame from relief model.
    
    Strategy: Remove faces that form a rectangular frame around the edges.
    These faces are typically vertical (or near-vertical) walls.
    """
    print(f"  Trimming border frame from: {Path(input_stl).name}")
    
    mesh = trimesh.load(input_stl)
    
    # Get face normals
    face_normals = mesh.face_normals
    
    # Get bounding box
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Define edge threshold (faces within this distance of bbox edges)
    edge_threshold = min(bbox_size) * 0.05  # 5% of smallest dimension
    
    # Get face centers
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)
    
    # Identify edge faces (near bounding box edges)
    near_x_min = face_centers[:, 0] < (bbox_min[0] + edge_threshold)
    near_x_max = face_centers[:, 0] > (bbox_max[0] - edge_threshold)
    near_y_min = face_centers[:, 1] < (bbox_min[1] + edge_threshold)
    near_y_max = face_centers[:, 1] > (bbox_max[1] - edge_threshold)
    
    # Frame faces are near edges
    is_edge_face = near_x_min | near_x_max | near_y_min | near_y_max
    
    # Also check if face is vertical (normal pointing sideways, not up/down)
    # Vertical faces have low Z component in normal
    is_vertical = np.abs(face_normals[:, 2]) < 0.3  # Z component < 0.3
    
    # Frame faces are both near edges AND vertical
    is_frame_face = is_edge_face & is_vertical
    
    # Keep faces that are NOT frame faces
    keep_faces = ~is_frame_face
    
    # Create trimmed mesh
    trimmed_faces = mesh.faces[keep_faces]
    
    # Get unique vertices
    unique_verts = np.unique(trimmed_faces)
    vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_verts)}
    remapped_faces = np.array([[vert_map[v] for v in face] for face in trimmed_faces])
    
    trimmed_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[unique_verts],
        faces=remapped_faces,
        process=False
    )
    
    # Report
    original_faces = len(mesh.faces)
    removed = original_faces - len(trimmed_faces)
    removed_pct = (removed / original_faces) * 100
    
    print(f"    Removed {removed:,} faces ({removed_pct:.1f}%) - frame eliminated")
    print(f"    Kept {len(trimmed_faces):,} faces - relief surface + base")
    
    # Save
    output_stl = Path(output_stl)
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    trimmed_mesh.export(output_stl)
    
    return output_stl


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Trim border frame from relief STL")
    parser.add_argument("--input", required=True, help="Input STL with frame")
    parser.add_argument("--output", required=True, help="Output trimmed STL")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Z threshold ratio")
    
    args = parser.parse_args()
    
    try:
        trim_border_frame(args.input, args.output, args.tolerance)
        print("[OK] Border trimmed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Trimming failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)