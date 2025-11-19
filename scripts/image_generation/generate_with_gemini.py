#!/usr/bin/env python3
"""
generate_with_gemini.py - Standard-quality image generation using Vertex AI.
This script provides the "basic quality" option for the pipeline.

Usage:
  python generate_with_gemini.py --prompt "oak tree bust" --out ../2D_files/out.png

Notes:
 - This script uses the Vertex AI library, consistent with other AI scripts.
 - It uses the GOOGLE_CLOUD_PROJECT and 'gcloud auth' credentials.
 - It does NOT use a GEMINI_API_KEY.
"""

import os
import argparse
from pathlib import Path
import sys
import time
import threading
import warnings

# Suppress gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    # Import from the Vertex AI library
    import vertexai
    from vertexai.vision_models import ImageGenerationModel, Image
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    HAS_VERTEXAI = True
except Exception as e:
    HAS_VERTEXAI = False
    VERTEXAI_IMPORT_ERROR = e

# Prompt template (can be the same as Imagen 3's)
PROMPT_TEMPLATE = """
Grayscale, photorealistic, razor-sharp, ultra-detailed, single image suitable for bas relief and CNC cutout, designed to maximize perceived depth and dynamic form. {description}. The object(s) are presented in a perfectly clean manner to reveal maximal volumetric detail and reduce any distraction from the subject, place it prominently in the center, ensuring maximum detail.

Visual Style: Grayscale rendering, studio quality, 8K resolution, hyper-sharp focus throughout the entire object, with microscopic detail, meticulously rendered. The image should resemble a highly detailed heightmap or individual clay sculptures, with every surface texture, edge, and contour clearly defined. Lighting is controlled and even, designed to emphasize form through subtle gradients and shadows that enhance depth without creating harsh contrasts. The tonal range is optimized to capture fine elevation changes, with deep blacks within the object's form representing the deepest recesses and bright whites for the highest points.

Composition & Depth: The subject is isolated on a pure white background with no distractions or shadows. The viewing angle is carefully chosen to maximize the perception of three-dimensional form. If multiple elements are present, they should be arranged to showcase clear spatial relationships and depth layering. The framing is tight but not cropped, ensuring all important details are visible while maintaining a clear focus on the primary subject.

Technical Requirements: Perfect edge definition with no soft focus or depth-of-field blur. Uniform sharpness across the entire image plane. High contrast in surface detail while maintaining smooth tonal transitions. Clear separation between different depth planes. Optimal lighting to reveal all surface variations and contours. No atmospheric effects, haze, or environmental obscuration.
"""

# Negative prompt
NEGATIVE_PROMPT = """
multiple views, extreme perspective, heavy distortion, bokeh, atmospheric haze, motion blur, color, sepia, tinted backgrounds, text, watermarks, logos, busy patterns, cluttered composition, soft focus, depth of field, vignetting, chromatic aberration, noise, grain, artifacts, compression, low resolution, blurry edges, flat lighting, overexposure, underexposure, harsh shadows, multiple light sources, reflections, lens flare
"""


class ProgressIndicator:
    """Shows a simple progress indicator with elapsed time during long operations."""
    def __init__(self, message="Processing"):
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None
        
    def _animate(self):
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while self.running:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            sys.stdout.write(f"\r  {spinner[idx]} {self.message} [{time_str}]")
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner)
            time.sleep(0.2)
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self, success_msg=None):
        self.running = False
        if self.thread:
            self.thread.join()
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        if success_msg:
            sys.stdout.write(f"\r  [OK] {success_msg} [{time_str}]" + " " * 20 + "\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

def check_credentials():
    """
    Check if credentials are properly configured.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "ERROR: GOOGLE_CLOUD_PROJECT environment variable not set.\n"
            "="*60 + "\n\n"
            "This script uses Vertex AI and requires a Google Cloud Project."
        )
    
    try:
        credentials, detected_project = google.auth.default()
        return credentials, project_id
    except DefaultCredentialsError:
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "ERROR: Could not find Google Cloud credentials.\n"
            "="*60 + "\n\n"
            "Run 'gcloud auth application-default login' to authenticate."
        )


def generate_image(prompt_text: str, output_filename: str):
    """Generates an image using a standard Imagen model on Vertex AI."""
    
    if not HAS_VERTEXAI:
        raise RuntimeError(f"Vertex AI SDK not installed: {VERTEXAI_IMPORT_ERROR}")

    try:
        credentials, project_id = check_credentials()
    except RuntimeError as e:
        print(str(e))
        return False
    
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    
    print(f"Initializing Imagen (Basic) on Vertex AI...")
    print(f"  Project: {project_id}, Region: {location}")
    print(f"  Description: {prompt_text[:60]}...")
    
    progress = ProgressIndicator("Generating image with Imagen (Vertex AI)")
    progress.start()
    
    try:
        vertexai.init(project=project_id, location=location, credentials=credentials)
        
        # --- THIS IS THE KEY CHANGE ---
        # Load the standard, "basic" image generation model
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        
        # Create the final prompt
        final_prompt = PROMPT_TEMPLATE.format(description=prompt_text)
        
        # Generate the image
        images = model.generate_images(
            prompt=final_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            number_of_images=1,
            aspect_ratio="1:1",
            # Safety settings for this model version
            # safety_filter_level="block_some", 
            # person_generation="allow_adult",
        )
        
        if images and len(images.images) > 0:
            image = images.images[0]
            image.save(location=output_filename)
            
            progress.stop("Image generated successfully")
            return True
        else:
            progress.stop()
            print("[X] Failed to generate image. No images returned.")
            return False
            
    except Exception as e:
        progress.stop()
        print(f"\n[X] Error during image generation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate images using a standard Imagen model on Vertex AI")
    
    parser.add_argument("--prompt", required=True, help="Description of what to generate")
    parser.add_argument("--out", required=True, help="Output file path")
    args = parser.parse_args()

    try:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        outp = generate_image(args.prompt, args.out)
        
        if not outp:
             raise RuntimeError("Image generation failed (see error above)")
             
        print(f"\nSaved to: {args.out}")
    except Exception as e:
        print(f"\n[ERROR] Image generation failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()