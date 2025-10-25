#!/usr/bin/env python3
"""
Preprocess cropped segments for better Marigold depth estimation.
Adds context and upscales for maximum detail.
"""
from PIL import Image, ImageFilter
import numpy as np


def add_context_padding(cropped_img, padding_percent=10, blur_radius=20):
    """
    Add blurred padding around cropped segment to give Marigold context.
    
    Args:
        cropped_img: PIL Image of cropped segment
        padding_percent: % of image size to add as padding
        blur_radius: How much to blur the padding
    
    Returns:
        PIL Image with context padding
    """
    w, h = cropped_img.size
    
    # Calculate padding size
    pad_w = int(w * padding_percent / 100)
    pad_h = int(h * padding_percent / 100)
    
    # Create canvas with padding
    new_w = w + 2 * pad_w
    new_h = h + 2 * pad_h
    
    # Fill with blurred/averaged version of edges
    canvas = Image.new('RGB', (new_w, new_h), color=(128, 128, 128))
    
    # Paste original in center
    canvas.paste(cropped_img, (pad_w, pad_h))
    
    # Blur the entire image
    blurred = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Paste sharp original back in center
    blurred.paste(cropped_img, (pad_w, pad_h))
    
    return blurred


def preprocess_segment(input_path, output_path, config):
    """
    Full preprocessing pipeline for cropped segments.
    """
    img = Image.open(input_path).convert('RGB')
    
    print(f"Preprocessing segment: {img.size}")
    
    # Step 1: Add context padding
    if config.get('add_context_padding', True):
        padding = config.get('padding_percent', 10)
        print(f"  Adding {padding}% context padding...")
        img = add_context_padding(img, padding)
    
    # Step 2: Upscale
    if config.get('upscale_before_depth', True):
        from ai_upscale import upscale_image
        
        scale = config.get('upscale_factor', 4)
        method = config.get('upscale_method', 'auto')
        
        temp_upscaled = output_path.parent / f"{output_path.stem}_upscaled.png"
        img.save(temp_upscaled)
        
        upscale_image(temp_upscaled, output_path, scale, method)
        temp_upscaled.unlink()
    else:
        img.save(output_path)
    
    print(f"  Preprocessed segment saved: {Image.open(output_path).size}")
    
    return output_path