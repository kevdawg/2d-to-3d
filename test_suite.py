#!/usr/bin/env python3
"""
Automated Testing Suite for 2D-to-3D Pipeline
Tests multiple settings on reference images to validate quality and find optimal settings.

Usage:
    python test_suite.py --test all
    python test_suite.py --test depth --image test_images/portrait.jpg
    python test_suite.py --compare settings1.yaml settings2.yaml
"""
import argparse
from pathlib import Path
import yaml
import time
import json
import numpy as np
from PIL import Image
import cv2


class QualityMetrics:
    """Calculate quality metrics for depth maps."""
    
    @staticmethod
    def calculate_depth_consistency(depth_array):
        """
        Measure how smooth/consistent the depth map is.
        Lower = smoother, higher = more variation.
        """
        # Calculate gradients
        grad_x = np.gradient(depth_array, axis=1)
        grad_y = np.gradient(depth_array, axis=0)
        
        # Total variation (smoothness measure)
        tv = np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y))
        
        return float(tv)
    
    @staticmethod
    def calculate_detail_preservation(depth_array):
        """
        Measure how much fine detail is preserved.
        Uses high-frequency content as proxy.
        """
        # Laplacian (edge detection)
        laplacian = cv2.Laplacian(depth_array.astype(np.float32), cv2.CV_64F)
        detail_score = np.std(laplacian)
        
        return float(detail_score)
    
    @staticmethod
    def calculate_noise_level(depth_array):
        """
        Estimate noise level using high-frequency variance.
        Higher = more noise.
        """
        # High-pass filter to isolate noise
        blur = cv2.GaussianBlur(depth_array.astype(np.float32), (5, 5), 0)
        noise = depth_array - blur
        noise_level = np.std(noise)
        
        return float(noise_level)
    
    @staticmethod
    def calculate_dynamic_range(depth_array):
        """
        Measure the range of depth values used.
        Higher = more depth variation (more 3D-looking).
        """
        min_val = np.min(depth_array)
        max_val = np.max(depth_array)
        
        # Normalize to 0-1 range
        if max_val > min_val:
            dynamic_range = (max_val - min_val) / 65535.0
        else:
            dynamic_range = 0.0
        
        return float(dynamic_range)
    
    @staticmethod
    def analyze_depth_map(depth_path):
        """
        Complete analysis of a depth map.
        
        Returns:
            dict with all metrics
        """
        depth_img = Image.open(depth_path)
        depth_array = np.array(depth_img)
        
        # Ensure 2D
        if depth_array.ndim > 2:
            depth_array = depth_array[:, :, 0]
        
        metrics = {
            'consistency': QualityMetrics.calculate_depth_consistency(depth_array),
            'detail': QualityMetrics.calculate_detail_preservation(depth_array),
            'noise': QualityMetrics.calculate_noise_level(depth_array),
            'dynamic_range': QualityMetrics.calculate_dynamic_range(depth_array),
        }
        
        # Overall quality score (weighted combination)
        # Higher detail and range are good, lower noise and moderate consistency are good
        quality_score = (
            metrics['detail'] * 0.4 +
            metrics['dynamic_range'] * 100 * 0.3 -
            metrics['noise'] * 0.2 +
            (1.0 / (1.0 + metrics['consistency'])) * 0.1
        )
        
        metrics['quality_score'] = quality_score
        
        return metrics


class SettingsComparator:
    """Compare results from different settings."""
    
    def __init__(self, reference_images_dir):
        self.reference_dir = Path(reference_images_dir)
        self.results = {}
    
    def test_settings(self, image_path, settings_name, config):
        """
        Run pipeline with specific settings and measure quality.
        
        Args:
            image_path: Path to test image
            settings_name: Name for this test (e.g., "high_quality", "custom_1")
            config: Config dict with settings to test
        
        Returns:
            dict with metrics and timing
        """
        print(f"\nTesting: {settings_name}")
        print(f"Image: {Path(image_path).name}")
        
        start_time = time.time()
        
        # TODO: Run actual pipeline here
        # For now, placeholder
        # depth_path = run_pipeline(image_path, config)
        
        # Placeholder - load existing depth if available
        depth_path = Path(image_path).parent / f"{Path(image_path).stem}_depth_16bit.png"
        
        if not depth_path.exists():
            print(f"  ⚠️  Depth map not found: {depth_path}")
            return None
        
        elapsed = time.time() - start_time
        
        # Analyze quality
        metrics = QualityMetrics.analyze_depth_map(depth_path)
        metrics['processing_time'] = elapsed
        metrics['settings_name'] = settings_name
        
        print(f"  Quality Score: {metrics['quality_score']:.2f}")
        print(f"  Detail: {metrics['detail']:.2f}")
        print(f"  Dynamic Range: {metrics['dynamic_range']:.2f}")
        print(f"  Noise: {metrics['noise']:.2f}")
        print(f"  Processing Time: {elapsed:.1f}s")
        
        return metrics
    
    def compare_settings(self, image_path, settings_list):
        """
        Test multiple settings on same image and compare.
        
        Args:
            image_path: Path to test image
            settings_list: List of (name, config) tuples
        
        Returns:
            Comparison report dict
        """
        results = []
        
        for name, config in settings_list:
            metrics = self.test_settings(image_path, name, config)
            if metrics:
                results.append(metrics)
        
        # Rank by quality score
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print("\n" + "="*60)
        print("  COMPARISON RESULTS")
        print("="*60)
        print(f"\nRanking (best to worst):")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['settings_name']}: {result['quality_score']:.2f}")
        
        return results
    
    def save_report(self, results, output_path):
        """Save comparison report as JSON."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Report saved: {output_path}")


class ReferenceTests:
    """Run tests on reference image collection."""
    
    def __init__(self, reference_dir="test_images"):
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(exist_ok=True)
    
    def setup_reference_images(self):
        """
        Set up reference image collection for testing.
        Should include variety of subjects and scenarios.
        """
        categories = {
            'portraits': "Photos with human faces",
            'animals': "Animals (pets, wildlife)",
            'architecture': "Buildings and structures",
            'nature': "Landscapes and outdoor scenes",
            'objects': "Everyday objects and products",
            'text': "Images with text or logos",
        }
        
        for category, description in categories.items():
            cat_dir = self.reference_dir / category
            cat_dir.mkdir(exist_ok=True)
            
            readme = cat_dir / "README.txt"
            if not readme.exists():
                readme.write_text(f"Place {description.lower()} here for testing.")
        
        print(f"Reference image structure created at: {self.reference_dir}")
        print("\nAdd test images to each category folder:")
        for category in categories:
            print(f"  - {self.reference_dir / category}/")
    
    def run_category_tests(self, category, settings_name, config):
        """Run tests on all images in a category."""
        cat_dir = self.reference_dir / category
        
        if not cat_dir.exists():
            print(f"Category not found: {category}")
            return []
        
        images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
        
        if not images:
            print(f"No images found in: {cat_dir}")
            return []
        
        results = []
        for img in images:
            metrics = SettingsComparator(self.reference_dir).test_settings(
                img, settings_name, config
            )
            if metrics:
                metrics['category'] = category
                metrics['image'] = img.name
                results.append(metrics)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Automated testing suite for 2D-to-3D pipeline"
    )
    
    parser.add_argument("--test", choices=['all', 'depth', 'regional', 'settings'],
                       help="What to test")
    parser.add_argument("--image", help="Specific image to test")
    parser.add_argument("--compare", nargs='+', help="Compare multiple settings files")
    parser.add_argument("--setup", action='store_true', 
                       help="Set up reference image structure")
    
    args = parser.parse_args()
    
    if args.setup:
        ref_tests = ReferenceTests()
        ref_tests.setup_reference_images()
        return
    
    if args.compare:
        # Compare different settings
        print("Settings comparison not yet implemented")
        # TODO: Load configs from files and compare
    
    elif args.test:
        print(f"Running {args.test} tests...")
        # TODO: Implement specific tests
    
    else:
        print("No action specified. Use --help for usage.")


if __name__ == "__main__":
    main()