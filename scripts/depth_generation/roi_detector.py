#!/usr/bin/env python3
"""
ROI (Region of Interest) Detector
Detects faces using OpenCV and creates precise masks using SAM.
Models are downloaded automatically on first use.
"""
import cv2
import numpy as np
from pathlib import Path
import os
import requests
from tqdm import tqdm


class ROIDetector:
    """Detects regions of interest (primarily faces) in images."""
    
    def __init__(self, sam_checkpoint=None, use_sam=True, detection_mode='human', grounding_dino_model=None):
        """
        Initialize detector with optional SAM and Grounding DINO.
        
        Args:
            sam_checkpoint: Path to SAM model (auto-downloads if None)
            use_sam: If False, only returns bounding boxes (faster)
            detection_mode: 'human', 'automatic', 'contour', 'prompt'
            grounding_dino_model: Path to Grounding DINO model (auto-downloads if None)
        """
        self.use_sam = use_sam
        self.detection_mode = detection_mode
        
        # Initialize OpenCV cascades based on mode
        self.cascades = []
        
        if detection_mode in ['human', 'all']:
            # Human frontal faces
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if not cascade.empty():
                self.cascades.append(('human_face', cascade))
        
        if detection_mode in ['human', 'all']:
            # Human profile faces
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            if not cascade.empty():
                self.cascades.append(('human_profile', cascade))
        
        if len(self.cascades) == 0 and detection_mode not in ['automatic', 'contour', 'prompt']:
            print("⚠️  Warning: No detection cascades loaded!")
            print("   Using alternative detection method...")
        
        # Initialize Grounding DINO if prompt mode
        self.grounding_dino = None
        if detection_mode == 'prompt':
            self.grounding_dino = self._load_grounding_dino(grounding_dino_model)
        
        # Initialize SAM if requested
        self.sam = None
        if use_sam:
            self.sam = self._load_sam(sam_checkpoint)
    
    def _load_sam(self, checkpoint_path):
        """Load SAM model, downloading if necessary."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not installed. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        # Determine model path
        if checkpoint_path is None:
            # Default location in models folder
            script_dir = Path(__file__).resolve().parent
            models_dir = script_dir.parent.parent / "models" / "sam_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = models_dir / "sam_vit_h_4b8939.pth"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        # Download if not exists
        if not checkpoint_path.exists():
            print(f"\n📦 Downloading SAM model (~2.4GB)...")
            print(f"   This is a one-time download.")
            self._download_sam_model(checkpoint_path)
        
        # Load model
        print(f"Loading SAM model from: {checkpoint_path.name}")
        sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint_path))
        
        # Move to GPU if available
        device = "cuda" if self._has_cuda() else "cpu"
        sam.to(device=device)
        print(f"   SAM loaded on: {device}")
        
        return SamPredictor(sam)
    
    def _has_cuda(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _download_sam_model(self, output_path):
        """Download SAM model weights from official source."""
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc="Downloading SAM",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"✅ SAM model saved to: {output_path}")
    
    def _load_grounding_dino(self, model_path=None):
        """Load Grounding DINO model for prompt-based detection."""
        try:
            from groundingdino.util.inference import load_model
            import torch
        except ImportError:
            print("⚠️  Grounding DINO not installed!")
            print("   Install with: pip install groundingdino-py")
            print("   Falling back to contour detection...")
            return None
        
        # Determine model path
        if model_path is None:
            script_dir = Path(__file__).resolve().parent
            models_dir = script_dir.parent.parent / "models" / "grounding_dino"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            config_path = models_dir / "GroundingDINO_SwinT_OGC.py"
            checkpoint_path = models_dir / "groundingdino_swint_ogc.pth"
            
            # Download if not exists
            if not checkpoint_path.exists():
                print("\n📦 Downloading Grounding DINO model (~600MB)...")
                print("   This is a one-time download.")
                self._download_grounding_dino(models_dir, config_path, checkpoint_path)
        else:
            checkpoint_path = Path(model_path)
            config_path = checkpoint_path.parent / "GroundingDINO_SwinT_OGC.py"
        
        # Load model
        print(f"Loading Grounding DINO model...")
        try:
            model = load_model(str(config_path), str(checkpoint_path))
            
            # Move to GPU if available
            device = "cuda" if self._has_cuda() else "cpu"
            model = model.to(device)
            print(f"   Grounding DINO loaded on: {device}")
            
            return model
        except Exception as e:
            print(f"⚠️  Failed to load Grounding DINO: {e}")
            print("   Falling back to contour detection...")
            return None

    def _download_grounding_dino(self, models_dir, config_path, checkpoint_path):
        """Download Grounding DINO model and config."""
        import requests
        from tqdm import tqdm
        
        # Download config
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        print("   Downloading config...")
        response = requests.get(config_url)
        config_path.write_bytes(response.content)
        
        # Download checkpoint
        checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        print("   Downloading checkpoint (~600MB)...")
        
        response = requests.get(checkpoint_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(checkpoint_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"✅ Grounding DINO model saved to: {models_dir}")

    def detect_with_prompt(self, image_path, text_prompt, confidence_threshold=0.25, min_size_percent=5.0):
        """Detect objects using text prompt with Grounding DINO."""
        if self.grounding_dino is None:
            print("   ⚠️  Grounding DINO not available, falling back to contour detection")
            return self.detect_subjects_contour(image_path, min_size_percent)
        
        try:
            from groundingdino.util.inference import predict
            import torch
            from PIL import Image as PILImage
        except ImportError:
            print("   ⚠️  Grounding DINO dependencies missing")
            return self.detect_subjects_contour(image_path, min_size_percent)
        
        print(f"   Using Grounding DINO with prompt: '{text_prompt}'")
        
        # Load image
        img_pil = PILImage.open(image_path).convert("RGB")
        img_cv = cv2.imread(str(image_path))
        h, w = img_cv.shape[:2]
        img_area = h * w
        min_area = img_area * (min_size_percent / 100.0)
        
        # Run detection
        boxes, logits, phrases = predict(
            model=self.grounding_dino,
            image=img_pil,
            caption=text_prompt,
            box_threshold=confidence_threshold,
            text_threshold=0.25
        )
        
        print(f"   Found {len(boxes)} detection(s)")
        
        # Convert boxes from normalized [0-1] to pixel coordinates
        detected_boxes = []
        
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            # Convert from center format to corner format
            cx, cy, box_w, box_h = box.cpu().numpy()
            
            # Denormalize
            cx *= w
            cy *= h
            box_w *= w
            box_h *= h
            
            # Convert to (x, y, w, h) format
            x = int(cx - box_w / 2)
            y = int(cy - box_h / 2)
            w_box = int(box_w)
            h_box = int(box_h)
            
            # Clip to image bounds
            x = max(0, x)
            y = max(0, y)
            w_box = min(w - x, w_box)
            h_box = min(h - y, h_box)
            
            # Check size
            area = w_box * h_box
            area_percent = (area / img_area) * 100
            
            if area < min_area:
                print(f"   Rejected '{phrase}': {area_percent:.1f}% of image (too small)")
                continue
            
            confidence = float(logit)
            print(f"   Accepted '{phrase}': {area_percent:.1f}% of image, confidence: {confidence:.2f}")
            
            detected_boxes.append((x, y, w_box, h_box))
        
        # Remove overlapping boxes
        if len(detected_boxes) > 1:
            detected_boxes = self._remove_overlapping_detections(detected_boxes, overlap_threshold=0.3)
        
        print(f"   Final detections: {len(detected_boxes)}")
        
        return detected_boxes

    def detect_subjects_automatic(self, image_path, min_size_percent=5.0):
        """
        Automatically detect all significant subjects using SAM's automatic mask generation.
        Works for ANY subject (humans, animals, objects) without pre-trained cascades.
        
        WARNING: Very memory intensive! Requires 8GB+ RAM.
        For lower memory, use detect_subjects_contour() instead.
        
        Args:
            image_path: Path to input image
            min_size_percent: Minimum subject size as % of image (default: 5%)
        
        Returns:
            List of bounding boxes for detected subjects
        """
        if not self.use_sam or self.sam is None:
            print("   ⚠️  SAM required for automatic detection")
            print("   Falling back to contour-based detection...")
            return self.detect_subjects_contour(image_path, min_size_percent)
        
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError:
            print("   ⚠️  SamAutomaticMaskGenerator not available")
            return self.detect_subjects_contour(image_path, min_size_percent)
        
        print("   Using SAM automatic mask generation (this may take a while and use lots of RAM)...")
        
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img_area = h * w
        
        # Reduce image size to save memory if too large
        max_dimension = 1024
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            print(f"   Resized to {new_w}x{new_h} to save memory")
        
        try:
            # Create automatic mask generator with lower settings for memory
            mask_generator = SamAutomaticMaskGenerator(
                model=self.sam.model,
                points_per_side=16,  # Reduced from 32 to save memory
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=100,
            )
            
            # Generate masks
            masks = mask_generator.generate(img_rgb)
            
            print(f"   Generated {len(masks)} candidate masks")
            
            # Convert masks to bounding boxes and filter
            min_area = img_area * (min_size_percent / 100.0)
            boxes = []
            
            for mask_data in masks:
                area = mask_data['area']
                
                # Skip tiny regions
                if area < min_area * 0.1:  # Even more lenient during generation
                    continue
                
                # Get bounding box and scale back if needed
                bbox = mask_data['bbox']  # [x, y, w, h]
                if max(h, w) > max_dimension:
                    scale_back = max(h, w) / max_dimension
                    bbox = [int(b * scale_back) for b in bbox]
                
                boxes.append(tuple(bbox))
            
            print(f"   Filtered to {len(boxes)} significant subjects")
            return boxes
            
        except Exception as e:
            print(f"   ⚠️  SAM automatic failed (out of memory?): {e}")
            print("   Falling back to contour-based detection...")
            return self.detect_subjects_contour(image_path, min_size_percent)
    
    def detect_subjects_contour(self, image_path, min_size_percent=5.0):
        """
        Lightweight subject detection using edge/contour detection.
        Works well for images with background already removed.
        Low memory usage, works on any hardware.
        
        Args:
            image_path: Path to input image
            min_size_percent: Minimum subject size as % of image (default: 5%)
        
        Returns:
            List of bounding boxes for detected subjects
        """
        print("   Using contour-based detection (lightweight, works for any subject)...")
        
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = img.shape[:2]
        img_area = h * w
        min_area = img_area * (min_size_percent / 100.0)
        
        # Check if image has alpha channel (background removed)
        if img.shape[2] == 4:
            # Use alpha channel to find foreground (subject)
            alpha = img[:, :, 3]
            # Foreground = alpha > 128 (opaque pixels = subject)
            mask = (alpha > 128).astype(np.uint8) * 255
            print("   Using alpha channel to detect subject (opaque = foreground)")
        else:
            # No alpha - use edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Threshold to find foreground
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print("   Using edge detection (no alpha channel)")
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Find contours (RETR_EXTERNAL = only outer contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size and get bounding boxes
        boxes = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            area_percent = (area / img_area) * 100
            
            if area < min_area:
                print(f"   Rejected contour {i}: {area_percent:.1f}% of image (too small)")
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            boxes.append((x, y, w_box, h_box))
            print(f"   Accepted contour {i}: {area_percent:.1f}% of image at ({x}, {y}), size {w_box}x{h_box}")
        
        # Merge overlapping boxes
        if len(boxes) > 1:
            boxes = self._remove_overlapping_detections(boxes, overlap_threshold=0.3)
        
        print(f"   Found {len(boxes)} subject(s) via contours")
        
        return boxes
    
    def detect_faces(self, image_path, min_size_percent=5.0, scale_factor=1.05):
        """
        Detect faces/subjects in image using configured cascades with intelligent filtering.
        
        Args:
            image_path: Path to input image
            min_size_percent: Minimum detection size as % of image dimension (default: 5%)
            scale_factor: Detection scale factor (lower = more accurate but slower)
        
        Returns:
            List of detection bounding boxes: [(x, y, w, h), ...]
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        
        # Calculate minimum size based on image dimensions
        min_dimension = min(h, w)
        min_size = int(min_dimension * (min_size_percent / 100.0))
        min_size = max(min_size, 30)  # Never go below 30 pixels
        
        print(f"   Image size: {w}x{h}")
        print(f"   Minimum detection size: {min_size}x{min_size} ({min_size_percent}% of {min_dimension})")
        
        all_detections = []
        
        # Run all loaded cascades
        for cascade_name, cascade in self.cascades:
            detections = cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=8,  # Stricter to reduce false positives
                minSize=(min_size, min_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(detections) > 0:
                print(f"   {cascade_name}: {len(detections)} raw detection(s)")
                all_detections.extend(detections)
        
        if len(all_detections) == 0:
            return np.array([])
        
        # Remove duplicates (same region detected by multiple cascades)
        all_detections = self._remove_overlapping_detections(all_detections)
        
        # Filter out tiny detections and obvious false positives
        img_area = h * w
        filtered_faces = []
        
        for (x, y, w_box, h_box) in all_detections:
            face_area = w_box * h_box
            face_percent = (face_area / img_area) * 100
            
            # Skip if detection is less than 0.5% of image (likely false positive)
            if face_percent < 0.5:
                print(f"   Rejected tiny detection: {face_percent:.2f}% of image")
                continue
            
            # Skip extremely elongated boxes (likely not faces)
            aspect_ratio = w_box / h_box
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                print(f"   Rejected weird aspect ratio: {aspect_ratio:.2f}")
                continue
            
            filtered_faces.append((x, y, w_box, h_box))
        
        print(f"   Final detections: {len(filtered_faces)}")
        
        return np.array(filtered_faces) if filtered_faces else np.array([])
    
    def _remove_overlapping_detections(self, detections, overlap_threshold=0.5):
        """Remove duplicate detections that significantly overlap."""
        if len(detections) == 0:
            return []
        
        # Convert to (x1, y1, x2, y2) format
        boxes = []
        for (x, y, w, h) in detections:
            boxes.append([x, y, x + w, y + h])
        boxes = np.array(boxes)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by area (largest first)
        indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Keep the largest box
            i = indices[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / areas[indices[1:]]
            
            # Remove overlapping boxes
            indices = indices[1:][overlap < overlap_threshold]
        
        # Convert back to (x, y, w, h) format
        result = []
        for i in keep:
            x, y, x2, y2 = boxes[i]
            result.append((int(x), int(y), int(x2 - x), int(y2 - y)))
        
        return result
    
    def create_region_masks(self, image_path, detection_mode=None, faces=None, padding=0.2):
        """
        Create masks for detected regions using SAM for precision.
        
        Args:
            image_path: Path to input image
            detection_mode: 'human', 'automatic', or None (use self.detection_mode)
            faces: Pre-detected boxes (if None, detects automatically)
            padding: Padding around detected boxes (0.2 = 20% larger box)
        
        Returns:
            dict: {
                'faces': List of binary masks (one per detected subject),
                'background': Binary mask for non-subject regions,
                'face_boxes': List of (x, y, w, h) tuples
            }
        """
        # Load image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Determine detection mode
        if detection_mode is None:
            detection_mode = self.detection_mode
        
        # Detect subjects if not provided
        if faces is None:
            if detection_mode == 'prompt':
                # Get prompt from config
                text_prompt = getattr(self, 'detection_prompt', 'animal face. cow head. goat face')
                confidence = getattr(self, 'prompt_confidence', 0.25)
                faces = self.detect_with_prompt(image_path, text_prompt, confidence)
            elif detection_mode == 'automatic':
                faces = self.detect_subjects_automatic(image_path)
            elif detection_mode == 'contour':
                faces = self.detect_subjects_contour(image_path)
            elif detection_mode == 'human':
                faces = self.detect_faces(image_path)
            elif detection_mode == 'all':
                # Try human detection first
                faces = self.detect_faces(image_path)
                # If nothing found, try contour (not automatic - too much RAM)
                if len(faces) == 0:
                    print("   No human faces detected, trying contour detection...")
                    faces = self.detect_subjects_contour(image_path)
            else:
                raise ValueError(f"Unknown detection_mode: {detection_mode}")
        
        if len(faces) == 0:
            print("   No subjects detected - processing entire image as single region")
            return {
                'faces': [],
                'background': np.ones((h, w), dtype=bool),
                'face_boxes': []
            }
        
        # Expand boxes with padding
        padded_boxes = []
        for (x, y, w_box, h_box) in faces:
            pad_w = int(w_box * padding)
            pad_h = int(h_box * padding)
            x_pad = max(0, x - pad_w)
            y_pad = max(0, y - pad_h)
            w_pad = min(w - x_pad, w_box + 2 * pad_w)
            h_pad = min(h - y_pad, h_box + 2 * pad_h)
            padded_boxes.append((x_pad, y_pad, w_pad, h_pad))
        
        # Create masks
        face_masks = []
        
        if self.use_sam and self.sam is not None:
            # Use SAM for precise segmentation
            self.sam.set_image(img_rgb)
            
            for (x, y, w_box, h_box) in padded_boxes:
                # Convert to SAM box format [x1, y1, x2, y2]
                box = np.array([x, y, x + w_box, y + h_box])
                
                # Generate mask
                masks, scores, _ = self.sam.predict(
                    box=box,
                    multimask_output=False
                )
                
                face_masks.append(masks[0])  # Take best mask
        else:
            # Use simple rectangular masks (fast, no SAM needed)
            for (x, y, w_box, h_box) in padded_boxes:
                mask = np.zeros((h, w), dtype=bool)
                mask[y:y+h_box, x:x+w_box] = True
                face_masks.append(mask)
        
        # Create background mask (everything not in a face)
        background_mask = np.ones((h, w), dtype=bool)
        for face_mask in face_masks:
            background_mask &= ~face_mask
        
        return {
            'faces': face_masks,
            'background': background_mask,
            'face_boxes': padded_boxes
        }
    
    def visualize_regions(self, image_path, regions, output_path):
        """
        Save visualization of detected regions.
        
        Args:
            image_path: Path to input image
            regions: Output from create_region_masks()
            output_path: Where to save visualization
        """
        img = cv2.imread(str(image_path))
        img_vis = img.copy()
        
        # Draw face boxes
        for (x, y, w, h) in regions['face_boxes']:
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_vis, 'FACE', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Overlay face masks with transparency
        overlay = img.copy()
        for i, mask in enumerate(regions['faces']):
            overlay[mask] = [0, 255, 0]  # Green for faces
        
        # Blend
        img_vis = cv2.addWeighted(img_vis, 0.7, overlay, 0.3, 0)
        
        cv2.imwrite(str(output_path), img_vis)
        print(f"   Saved region visualization: {Path(output_path).name}")


# Convenience function for quick testing
def quick_detect(image_path, use_sam=True, visualize=True):
    """Quick test function to detect faces in an image."""
    detector = ROIDetector(use_sam=use_sam)
    
    print(f"\nDetecting regions in: {Path(image_path).name}")
    regions = detector.create_region_masks(image_path)
    
    print(f"   Found {len(regions['faces'])} face(s)")
    
    if visualize:
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_regions.jpg"
        detector.visualize_regions(image_path, regions, output_path)
    
    return regions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect regions of interest in images")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--no-sam", action='store_true', help="Skip SAM (use boxes only)")
    parser.add_argument("--visualize", action='store_true', help="Save visualization")
    
    args = parser.parse_args()
    
    quick_detect(args.input, use_sam=not args.no_sam, visualize=args.visualize)