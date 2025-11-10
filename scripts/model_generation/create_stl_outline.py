# create_stl_outline.py
# Usage: python preprocess_stl.py input.stl outline.dxf--offset 2.0

import trimesh
import argparse
from shapely.geometry import Polygon, MultiPoint, Point
import ezdxf


def compute_silhouette_polygon(mesh, z_tolerance=1e-6):
    # Project vertices onto XY plane
    verts = mesh.vertices.copy()
    points2d = verts[:, :2]
    # compute a concave hull / alpha shape approximation.
    # Simple heuristic: compute the 2D alpha shape by Delaunay + filter - but for simplicity, use shapely's buffer on MultiPoint
    mp = MultiPoint([tuple(p) for p in points2d])
    # create a tiny buffer to merge adjacent points
    merged = mp.buffer(0.0)
    try:
        hull = merged.convex_hull  # fallback convex hull if concave fails
    except Exception:
        hull = mp.convex_hull
    return hull

def offset_polygon(poly: Polygon, offset_mm: float):
    # Positive offset: outward
    return poly.buffer(offset_mm)

def export_polygon_to_dxf(poly: Polygon, filename):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    if poly.geom_type == 'Polygon':
        exterior = list(poly.exterior.coords)
        msp.add_lwpolyline(exterior, close=True)
        for interior in poly.interiors:
            msp.add_lwpolyline(list(interior.coords), close=True)
    elif poly.geom_type == 'MultiPolygon':
        for p in poly:
            exterior = list(p.exterior.coords)
            msp.add_lwpolyline(exterior, close=True)
    doc.saveas(filename)
    print("Saved DXF:", filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input STL/OBJ file')
    parser.add_argument('out_dxf', help='output outline DXF')
    parser.add_argument('--offset', type=float, default=2.0, help='outline offset in same units as mesh (mm)')
    args = parser.parse_args()

    mesh = trimesh.load_mesh(args.input, force='mesh')

    silhouette = compute_silhouette_polygon(simp)
    outline = offset_polygon(silhouette, args.offset)
    export_polygon_to_dxf(outline, args.out_dxf)

if __name__ == '__main__':
    main()
