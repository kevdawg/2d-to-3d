#!/usr/bin/env python3
"""
Depth Map Blending
Intelligently composite multiple regional depth maps with smooth transitions.
"""
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from scipy.ndimage import distance_transform_edt, gaussian_filter


class DepthBlender:
    """Blend multiple regional depth maps into a coherent final depth map."""
    
    def __init__(self, blend_width=20):
        """
        Initialize blender.
        
        Args:
            blend_width: Width of blending zone at region boundaries (pixels)
        """
        self.blend_width = blend_width
    
    def create_blend_weights(self, masks, blend_width=None):
        """
        Create smooth blending weights for each region using distance transforms.
        
        Args:
            masks: List of binary masks (one per region)
            blend_width: Override blend width for this operation
        
        Returns:
            List of float weight maps (0-1) for each region
        """
        if blend_width is None:
            blend_width = self.blend_width
        
        h, w = masks[0].shape
        weights = []
        
        for mask in masks:
            # Distance transform: how far each pixel is from mask edge
            distance = distance_transform_edt(mask)
            
            # Create smooth falloff at edges
            weight = np.clip(distance / blend_width, 0, 1)
            
            # Gaussian smoothing for even smoother transitions
            weight = gaussian_filter(weight, sigma=blend_width/4)
            
            weights.append(weight)
        
        # Normalize so weights sum to 1 at every pixel
        weight_sum = np.sum(weights, axis=0)
        weight_sum = np.where(weight_sum == 0, 1, weight_sum)  # Avoid division by zero
        
        weights = [w / weight_sum for w in weights]
        
        return weights
    
    def blend_depth_maps(self, depth_maps, masks, method='weighted', normalize=True):
        """
        Blend multiple depth maps using masks and smooth transitions.
        
        Args:
            depth_maps: List of depth map arrays (same size)
            masks: List of binary masks corresponding to each depth map
            method: 'weighted' (smooth blend) or 'priority' (regions override in order)
            normalize: Normalize final depth to 0-65535 range
        
        Returns:
            Blended depth map array
        """
        if len(depth_maps) != len(masks):
            raise ValueError("Must have same number of depth maps and masks")
        
        if len(depth_maps) == 0:
            raise ValueError("Need at least one depth map")
        
        h, w = depth_maps[0].shape[:2]
        
        # Ensure all depth maps are same size and 2D
        processed_depths = []
        for depth in depth_maps:
            if depth.ndim > 2:
                depth = depth[:, :, 0]
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            processed_depths.append(depth.astype(np.float32))
        
        if method == 'weighted':
            # Smooth blending using distance-based weights
            weights = self.create_blend_weights(masks)
            
            # Weighted sum
            result = np.zeros((h, w), dtype=np.float32)
            for depth, weight in zip(processed_depths, weights):
                result += depth * weight
        
        elif method == 'priority':
            # Priority-based: later regions override earlier ones in overlap areas
            result = processed_depths[0].copy()
            
            for depth, mask in zip(processed_depths[1:], masks[1:]):
                # Smooth transition at boundaries
                weight = distance_transform_edt(mask)
                weight = np.clip(weight / self.blend_width, 0, 1)
                weight = gaussian_filter(weight, sigma=self.blend_width/4)
                
                # Blend only where mask is active
                result = np.where(mask, 
                                 result * (1 - weight) + depth * weight,
                                 result)
        
        else:
            raise ValueError(f"Unknown blend method: {method}")
        
        # Normalize to 16-bit range
        if normalize:
            min_val, max_val = np.nanmin(result), np.nanmax(result)
            if max_val > min_val:
                result = ((result - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
            else:
                result = np.zeros_like(result, dtype=np.uint16)
        
        return result
    
    def visualize_blend(self, depth_maps, masks, output_path):
        """
        Create visualization showing how regions are blended.
        
        Args:
            depth_maps: List of depth maps
            masks: List of masks
            output_path: Where to save visualization
        """
        weights = self.create_blend_weights(masks)
        
        # Create RGB visualization (each region gets a color)
        colors = [
            [255, 0, 0],      # Red - Face 1
            [0, 255, 0],      # Green - Face 2
            [0, 0, 255],      # Blue - Face 3
            [255, 255, 0],    # Yellow - Face 4
            [128, 128, 128],  # Gray - Background
        ]
        
        h, w = masks[0].shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, (weight, color) in enumerate(zip(weights, colors[:len(weights)])):
            for c in range(3):
                vis[:, :, c] += (weight * color[c]).astype(np.uint8)
        
        Image.fromarray(vis).save(output_path)
        print(f"   Saved blend visualization: {Path(output_path).name}")


# Convenience function
def quick_blend(depth_paths, mask_arrays, output_path, method='weighted'):
    """Quick function to blend depth maps."""
    blender = DepthBlender(blend_width=30)
    
    # Load depth maps
    depth_maps = []
    for path in depth_paths:
        depth_img = Image.open(path)
        depth_array = np.array(depth_img)
        depth_maps.append(depth_array)
    
    # Blend
    result = blender.blend_depth_maps(depth_maps, mask_arrays, method=method)
    
    # Save
    Image.fromarray(result, mode='I;16').save(output_path)
    print(f"Blended depth saved: {output_path}")
    
    return result


if __name__ == "__main__":
    print("Depth blender module - import and use in pipeline")