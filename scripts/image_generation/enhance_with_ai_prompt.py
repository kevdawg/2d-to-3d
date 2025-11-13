#!/usr/bin/env python3
"""
High-quality, AI-prompted image enhancement using Google's Imagen 3.
This script performs image-to-image editing to fix blur, add detail,
or change elements based on a text prompt.

Usage:
  python enhance_with_ai_prompt.py --input "path/to/blurry.jpg" --output "path/to/fixed.png" --prompt "fix blur" "sharpen the fur" --project "my-gcp-project-id"
"""
import os
import sys
import argparse
import warnings
from pathlib import Path

# Suppress gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    # --- FIX: Correct import path ---
    # We are no longer using '.preview'
    from vertexai.vision_models import ImageGenerationModel, Image
    # --- END FIX ---
    
    import vertexai
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    from PIL import Image as PILImage
except ImportError as e:
    print("[X] ERROR: Missing required library.")
    print(f"    Details: {e}")
    print("    Please install: pip install google-cloud-aiplatform pillow")
    print("    Or activate the correct conda environment: conda activate aigen")
    sys.exit(1)


def ai_repair_image(input_path: str, output_path: str, prompt_list: list, project_id: str):
    """
    Uses Imagen 3's edit_image feature to enhance/repair an input image.
    """
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    
    # --- FIX: Join the list of words back into a single string ---
    prompt_text = " ".join(prompt_list)
    # --- END FIX ---
    
    print(f"AI Repair Initialized...")
    print(f"  Project: {project_id}, Region: {location}")
    print(f"  Prompt: '{prompt_text}'")
    
    try:
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Load the Imagen 3 model
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        
        # Load the input image from file
        input_image = Image.load_from_file(input_path)
        
        print("  Sending image and prompt to AI for repair...")
        
        # "Leash" the AI. 0.0 = 100% original, 1.0 = 100% prompt.
        # 0.4 (40% prompt influence) is a good starting point for repair.
        prompt_strength_value = 0.4
        print(f"  Repair strength set to: {prompt_strength_value}")
        
        # Generate the image
        images = model.edit_image(
            base_image=input_image,  # <-- FIX: Use 'base_image'
            prompt=prompt_text,
            prompt_strength=prompt_strength_value, # <-- FIX: "Different Dog"
            number_of_images=1,
            safety_filter_level="block_some",
            person_generation="allow_adult",
        )
        
        if images and len(images.images) > 0:
            image = images.images[0]
            image.save(location=output_path)
            
            print(f"  [OK] Repaired image saved: {output_path}")
            return True
        else:
            print("[X] Failed to generate image. No images returned.", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"\n[X] Error during image repair: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Fix/enhance blurry images using Imagen 3 AI",
        epilog="Requires a valid Google Cloud Project ID with billing."
    )
    parser.add_argument("--input", required=True, help="Path to the blurry/input image")
    parser.add_argument("--output", required=True, help="Output path for the repaired image")
    
    # --- FIX: Use nargs='+' to accept multiple words for the prompt ---
    parser.add_argument("--prompt", required=True, nargs="+",
                        help="Instructions for the AI (e.g., 'fix blur sharpen fur')")
    # --- END FIX ---
    
    parser.add_argument("--project", required=True, help="Google Cloud Project ID")
    
    args = parser.parse_args()
    
    # Ensure input exists
    if not Path(args.input).exists():
        print(f"[X] Error: Input file not found at {args.input}", file=sys.stderr)
        sys.exit(1)
        
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 'args.prompt' is now a list of strings, which 'ai_repair_image' will join
    success = ai_repair_image(args.input, args.output, args.prompt, args.project)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()