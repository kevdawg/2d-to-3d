#!/usr/bin/env python3
"""
generate_with_gemini.py - FREE tier image generation using Gemini 2.0 Flash.

Usage:
  python generate_with_gemini.py --prompt "oak tree bust" --out ../2D_files/out.png

Notes:
 - Install SDK: pip install google-genai
 - Set GEMINI_API_KEY in your environment (setx GEMINI_API_KEY "KEY" for persistence).
 - Uses Gemini 2.0 Flash with native image generation (FREE tier).
"""

import os
import argparse
from pathlib import Path
import sys
import time
import threading

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except Exception as e:
    HAS_GENAI = False
    GENAI_IMPORT_ERROR = e

PROMPT_TEMPLATE = """Grayscale, photorealistic, **razor-sharp**, ultra-detailed, single image suitable for bas relief and CNC cutout, designed to maximize perceived depth and dynamic form. **{description} on a perfectly transparent background.** The object(s) are presented in a perfectly clean **manner**, captured from a strong side profile view with a subtle, controlled turn to reveal maximal volumetric detail. If there are multiple objects then every object must be fully visible and distinctly separated from all other objects and the image edges, with no overlap whatsoever. If there is a singular object, place it prominently in the center, ensuring maximum detail without excessive cropping.

**Visual Style:**
Grayscale rendering, studio quality, 8K resolution, **hyper-sharp focus throughout the entire object, with microscopic detail**, meticulously rendered. The image should resemble a highly detailed heightmap or individual clay sculptures, emphasizing volumetric form through subtle tonal variations from pure black to pure white. Every single surface contour, undulation, and intricate detail of each object must be clearly defined by precise changes in luminosity, creating an undeniable sense of three-dimensional depth within the two-dimensional image. Ensure **laser-sharp**, smoothly anti-aliased silhouettes and consistently rendered organic edges without abrupt terminations for all objects.

**Lighting:**
Optimized for **extreme depth perception and detail definition**: a single, strong, very shallow-angle directional key light (e.g., from top-left, almost skimming the surface) combined with subtle ambient occlusion and soft fill to illuminate recessed areas. The lighting should meticulously sculpt the forms of all objects, generating precise, high-contrast gradients of light and shadow that define every peak and valley with absolute clarity, making each individual object appear as if it has significant physical depth. Avoid harsh, blown-out highlights or completely black, featureless shadows; instead, aim for a full tonal range that emphasizes form.

**Background & Composition:**
All objects are individually isolated on a **perfectly transparent background.** The composition features a clean, arranged display of the separate objects, ensuring ample clear space around each one. No clipping or overlapping of any kind. Each object should be distinct and prominently displayed within its own conceptual space on the canvas, highlighting its iconic silhouette and volumetric shape for bas relief interpretation.

**Material & Texture:**
Extreme emphasis on **grayscale PBR (Physically Based Rendering) material definition and intricate surface detailing**. Textures must be ultra-high-frequency and rich in microscopic detail, showcasing every surface characteristic (e.g., wood grain, metallic sheen, fabric weave, minute surface imperfections, organic textures, individual hairs/fibers) primarily through variations in luminosity and contrast. The texture and detail should actively and precisely contribute to defining the contours and perceived depth across the entire form of each object.

**Negative Prompts:**
--no multiple views, extreme perspective, heavy distortion, bokeh, atmospheric haze, motion blur, color, sepia, tinted, harsh self-shadows, reflections, gradients, text, watermarks, branding, logo, artistic stylization, flat rendering, jagged edges, pixelation, background objects, unrealistic shading, no depth, 2d, cartoon, cropped, partial view, incorrect number of objects, identical objects, overlapping, touching, stacked, chaotic arrangement, unharmonious composition, **blurry, soft focus, low detail, smudged, blurry edges, lack of definition, dull lighting**
"""


class ProgressIndicator:
    """Shows a simple progress indicator with elapsed time during long operations."""
    def __init__(self, message="Processing"):
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None
        
    def _animate(self):
        # Simple ASCII spinner - works on all systems
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while self.running:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            # Use ASCII only
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
            # Clear line and print success with ASCII only
            sys.stdout.write(f"\r  [OK] {success_msg} [{time_str}]" + " " * 20 + "\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()


def save_image_bytes(image_bytes: bytes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return out_path


def sdk_generate_image(prompt_text: str, out_path: Path):
    """Generate image using Gemini 2.0 Flash (FREE tier)."""
    if not HAS_GENAI:
        raise RuntimeError(f"google-genai SDK not installed: {GENAI_IMPORT_ERROR}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Use: setx GEMINI_API_KEY \"YOUR_KEY\" and reopen CMD.")

    client = genai.Client(api_key=api_key)
    final_prompt = f"Generate an image: {PROMPT_TEMPLATE.format(description=prompt_text)}"
    
    progress = ProgressIndicator("Generating image with Gemini AI")
    progress.start()
    
    try:
        gen_config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=final_prompt,
            config=gen_config
        )
        
        # Extract image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    img_bytes = part.inline_data.data
                    result = save_image_bytes(img_bytes, out_path)
                    progress.stop(f"Image generated successfully ({len(img_bytes)//1024} KB)")
                    return result
        
        progress.stop()
        raise RuntimeError("No image data found in response")
        
    except Exception as e:
        progress.stop()
        raise RuntimeError(f"Image generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using Gemini 2.0 Flash (FREE)")
    parser.add_argument("--prompt", required=True, help="Description of what to generate")
    parser.add_argument("--out", required=True, help="Output file path")
    args = parser.parse_args()

    try:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        outp = sdk_generate_image(args.prompt, p)
        print(f"\nSaved to: {outp}")
    except Exception as e:
        print(f"\n[ERROR] Image generation failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()