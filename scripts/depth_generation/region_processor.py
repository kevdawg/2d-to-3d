#!/usr/bin/env python3
"""
Region-Specific Depth Map Processing
Process different regions with customized settings for optimal detail.
"""
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import subprocess
import cv2

# Add photo preprocessing to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent / "photo_preprocessing"))

from photo_preprocess import (
    denoise_nlm,
    enhance_contrast_clahe,
    sharpen_unsharp_mask,
    enhance_detail_laplacian,
    bilateral_filter_smooth
)


class RegionProcessor:
    """Process image regions with customized enhancement and depth settings."""
    
    def __init__(self, marigold_cli_path, conda_exe="conda", marigold_env="marigold"):
        """
        Initialize region processor.
        
        Args:
            marigold_cli_path: Path to marigold_cli.py
            conda_exe: Path to conda executable
            marigold_env: Name of conda environment for Marigold
        """
        self.marigold_cli = Path(marigold_cli_path)
        self.conda_exe = conda_exe
        self.marigold_env = marigold_env
        
        if not self.marigold_cli.exists():
            raise FileNotFoundError(f"Marigold CLI not found: {self.marigold_cli}")
    
    def preprocess_region(self, image_array, mask, settings):
        """
        Apply preprocessing to a specific region.
        
        Args:
            image_array: Full image as numpy array (RGB)
            mask: Boolean mask for region
            settings: Dict with preprocessing parameters
        
        Returns:
            Preprocessed image array (same size as input)
        """
        result = image_array.copy()
        
        # Extract region
        region_pixels = image_array[mask]
        
        if len(region_pixels) == 0:
            return result
        
        # Apply preprocessing steps
        if settings.get('denoise_strength', 0) > 0:
            # Denoise only the region
            temp_img = np.zeros_like(image_array)
            temp_img[mask] = region_pixels
            denoised = denoise_nlm(temp_img, settings['denoise_strength'])
            result[mask] = denoised[mask]
        
        if settings.get('clahe_clip', 0) > 0:
            # CLAHE on region
            temp_img = result.copy()
            enhanced = enhance_contrast_clahe(temp_img, settings['clahe_clip'])
            result[mask] = enhanced[mask]
        
        if settings.get('enhance_details', 0) > 0:
            # Detail enhancement
            temp_img = result.copy()
            detailed = enhance_detail_laplacian(temp_img, settings['enhance_details'])
            result[mask] = detailed[mask]
        
        if settings.get('sharpen_percent', 0) > 0:
            # Sharpening
            temp_img = result.copy()
            sharpened = sharpen_unsharp_mask(
                temp_img,
                radius=settings.get('sharpen_radius', 2),
                percent=settings['sharpen_percent'],
                threshold=settings.get('sharpen_threshold', 3)
            )
            result[mask] = sharpened[mask]
        
        if settings.get('bilateral_filter', False):
            # Bilateral smoothing (for background)
            temp_img = result.copy()
            smoothed = bilateral_filter_smooth(
                temp_img,
                d=settings.get('bilateral_d', 9),
                sigma_color=settings.get('bilateral_sigma', 75),
                sigma_space=settings.get('bilateral_sigma', 75)
            )
            result[mask] = smoothed[mask]
        
        return result
    
    def generate_depth_for_region(self, image_path, output_path, marigold_settings):
        """
        Run Marigold depth estimation with specific settings.
        
        Args:
            image_path: Path to preprocessed region image
            output_path: Where to save depth map
            marigold_settings: Dict with Marigold parameters
        
        Returns:
            Path to generated depth map
        """
        cmd = [
            self.conda_exe, "run", "-n", self.marigold_env, "--no-capture-output",
            "python", str(self.marigold_cli),
            "--input", str(image_path),
            "--output", str(output_path),
            "--steps", str(marigold_settings.get('steps', 20)),
            "--ensemble", str(marigold_settings.get('ensemble', 5)),
            "--processing_res", str(marigold_settings.get('processing_res', 1024))
        ]
        
        if marigold_settings.get('match_input_res', False):
            cmd.append("--match_input_res")
        
        # Run Marigold
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Marigold failed: {result.stderr}")
        
        return Path(output_path)
    
    def process_region(
        self, 
        image_path, 
        mask, 
        region_name,
        preprocess_settings,
        marigold_settings,
        output_dir
    ):
        """
        Complete processing pipeline for a single region.
        
        Args:
            image_path: Path to original image
            mask: Boolean mask for this region
            region_name: Name for output files (e.g., "face_1", "background")
            preprocess_settings: Dict with preprocessing params
            marigold_settings: Dict with Marigold params
            output_dir: Directory for intermediate files
        
        Returns:
            Depth map array for this region
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        print(f"   Processing {region_name}...")
        
        # Preprocess region
        preprocessed = self.preprocess_region(img_array, mask, preprocess_settings)
        
        # Create masked version (set non-region to black for visualization)
        masked_img = preprocessed.copy()
        masked_img[~mask] = 0
        
        # Save preprocessed region
        region_img_path = output_dir / f"{region_name}_preprocessed.png"
        Image.fromarray(masked_img).save(region_img_path)
        
        # Generate depth map
        depth_path = output_dir / f"{region_name}_depth.png"
        self.generate_depth_for_region(region_img_path, depth_path, marigold_settings)
        
        # Load depth map
        depth_img = Image.open(depth_path)
        depth_array = np.array(depth_img)
        
        # Ensure 2D
        if depth_array.ndim > 2:
            depth_array = depth_array[:, :, 0]
        
        return depth_array


if __name__ == "__main__":
    print("Region processor module - import and use in pipeline")