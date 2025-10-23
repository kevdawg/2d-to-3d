#!/usr/bin/env python3
"""
High-quality image generation using Google's Imagen 3 via Vertex AI.
Requires: pip install google-cloud-aiplatform
Authentication: Run 'gcloud auth application-default login' first

See CREDENTIALS_SETUP.md for detailed setup instructions.
"""
import os
import sys

# Suppress gRPC warnings before any Google imports
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import argparse
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings



try:
    from vertexai.preview.vision_models import ImageGenerationModel
    import vertexai
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
except ImportError as e:
    print("[X] ERROR: Missing required library.")
    print("    Please install: pip install google-cloud-aiplatform")
    print("\n    Or activate the correct conda environment:")
    print("    conda activate imagen")
    sys.exit(1)

# Detailed prompt template optimized for depth perception
PROMPT_TEMPLATE = """
Grayscale, photorealistic, razor-sharp, ultra-detailed, single image suitable for bas relief and CNC cutout, designed to maximize perceived depth and dynamic form. {description}. The object(s) are presented in a perfectly clean manner to reveal maximal volumetric detail and reduce any distraction from the subject, place it prominently in the center, ensuring maximum detail.

Visual Style: Grayscale rendering, studio quality, 8K resolution, hyper-sharp focus throughout the entire object, with microscopic detail, meticulously rendered. The image should resemble a highly detailed heightmap or individual clay sculptures, with every surface texture, edge, and contour clearly defined. Lighting is controlled and even, designed to emphasize form through subtle gradients and shadows that enhance depth without creating harsh contrasts. The tonal range is optimized to capture fine elevation changes, with deep blacks within the object's form representing the deepest recesses and bright whites for the highest points.

Composition & Depth: The subject is isolated on a pure white background with no distractions or shadows. The viewing angle is carefully chosen to maximize the perception of three-dimensional form. If multiple elements are present, they should be arranged to showcase clear spatial relationships and depth layering. The framing is tight but not cropped, ensuring all important details are visible while maintaining a clear focus on the primary subject.

Technical Requirements: Perfect edge definition with no soft focus or depth-of-field blur. Uniform sharpness across the entire image plane. High contrast in surface detail while maintaining smooth tonal transitions. Clear separation between different depth planes. Optimal lighting to reveal all surface variations and contours. No atmospheric effects, haze, or environmental obscuration.
"""

# Negative prompt to avoid unwanted elements
NEGATIVE_PROMPT = """
multiple views, extreme perspective, heavy distortion, bokeh, atmospheric haze, motion blur, color, sepia, tinted backgrounds, text, watermarks, logos, busy patterns, cluttered composition, soft focus, depth of field, vignetting, chromatic aberration, noise, grain, artifacts, compression, low resolution, blurry edges, flat lighting, overexposure, underexposure, harsh shadows, multiple light sources, reflections, lens flare
"""


def check_credentials():
    """
    Check if credentials are properly configured and provide helpful error messages.
    Returns: (credentials, project_id) or raises RuntimeError
    """
    # Check for project ID
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "ERROR: GOOGLE_CLOUD_PROJECT environment variable not set.\n"
            "="*60 + "\n\n"
            "Please set your Google Cloud project ID:\n\n"
            "Windows PowerShell:\n"
            '  $env:GOOGLE_CLOUD_PROJECT="your-project-id"\n'
            '  [System.Environment]::SetEnvironmentVariable(\'GOOGLE_CLOUD_PROJECT\', \'your-project-id\', \'User\')\n\n'
            "Windows CMD:\n"
            '  setx GOOGLE_CLOUD_PROJECT "your-project-id"\n\n'
            "Linux/Mac:\n"
            '  export GOOGLE_CLOUD_PROJECT="your-project-id"\n'
            '  echo \'export GOOGLE_CLOUD_PROJECT="your-project-id"\' >> ~/.bashrc\n\n'
            "See CREDENTIALS_SETUP.md for detailed instructions."
        )
    
    # Try to get default credentials
    try:
        credentials, detected_project = google.auth.default()
        return credentials, project_id
    except DefaultCredentialsError:
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "ERROR: Could not find Google Cloud credentials.\n"
            "="*60 + "\n\n"
            "You need to authenticate with Google Cloud:\n\n"
            "1. Install Google Cloud SDK:\n"
            "   https://cloud.google.com/sdk/docs/install\n\n"
            "2. Run authentication command:\n"
            "   gcloud auth application-default login\n\n"
            "3. Follow the browser prompts to sign in\n\n"
            "This will create credentials at:\n"
            "  Windows: %APPDATA%\\gcloud\\application_default_credentials.json\n"
            "  Linux/Mac: ~/.config/gcloud/application_default_credentials.json\n\n"
            "See CREDENTIALS_SETUP.md for detailed instructions."
        )


def generate_image(prompt_text: str, output_filename: str):
    """Generates an image using Imagen 3 via Vertex AI and saves it."""
    
    # Check credentials and get configuration
    try:
        credentials, project_id = check_credentials()
    except RuntimeError as e:
        print(str(e))
        return False
    
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    
    print(f"Generating image with Imagen 3 AI...")
    print(f"Project: {project_id}, Region: {location}")
    print(f"Description: {prompt_text}")
    
    try:
        # Initialize Vertex AI with credentials
        vertexai.init(project=project_id, location=location, credentials=credentials)
        
        # Load the Imagen 3 model
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        
        # Create the final prompt
        final_prompt = PROMPT_TEMPLATE.format(description=prompt_text)
        
        # Generate the image
        images = model.generate_images(
            prompt=final_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            number_of_images=1,
            aspect_ratio="1:1",  # Square for best results with depth conversion
            safety_filter_level="block_some",
            person_generation="allow_adult",
        )
        
        if images and len(images.images) > 0:
            # Save the first generated image
            image = images.images[0]
            image.save(location=output_filename)
            
            print(f"[OK] Image saved: {output_filename}")
            return True
        else:
            print("[X] Failed to generate image. No images returned.")
            return False
            
    except Exception as e:
        error_msg = str(e).lower()
        
        # Provide specific error messages for common issues
        if "permission" in error_msg or "403" in error_msg:
            print(
                "\n" + "="*60 + "\n"
                "ERROR: Permission denied.\n"
                "="*60 + "\n\n"
                "Possible causes:\n"
                "1. Vertex AI API not enabled for your project\n"
                "   Enable it at: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com\n\n"
                "2. Billing not set up for your project\n"
                "   Set up billing at: https://console.cloud.google.com/billing\n\n"
                "3. Insufficient permissions\n"
                "   Make sure your account has 'Vertex AI User' role\n"
            )
        elif "quota" in error_msg:
            print(
                "\n" + "="*60 + "\n"
                "ERROR: Quota exceeded.\n"
                "="*60 + "\n\n"
                "You've hit your API quota limit.\n"
                "Check your quotas at: https://console.cloud.google.com/iam-admin/quotas\n"
                "You may need to request a quota increase or wait for reset.\n"
            )
        elif "not found" in error_msg or "404" in error_msg:
            print(
                "\n" + "="*60 + "\n"
                "ERROR: Resource not found.\n"
                "="*60 + "\n\n"
                "Possible causes:\n"
                "1. Project ID is incorrect\n"
                f"   Current project: {project_id}\n"
                "   Verify at: https://console.cloud.google.com/\n\n"
                "2. Region not supported\n"
                f"   Current region: {location}\n"
                "   Try: us-central1 (default)\n"
            )
        else:
            print(f"\n[X] Error during image generation: {e}")
            print("\nTroubleshooting:")
            print("1. Verify authentication: gcloud auth list")
            print("2. Check project ID: gcloud config get-value project")
            print(f"3. Verify Vertex AI API is enabled for project: {project_id}")
            print("4. Ensure billing is enabled: https://console.cloud.google.com/billing")
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Google Imagen 3 via Vertex AI",
        epilog="See CREDENTIALS_SETUP.md for authentication setup instructions."
    )
    parser.add_argument(
        "--prompt", 
        required=True, 
        help="Description of the image to generate"
    )
    parser.add_argument(
        "--out", 
        required=True, 
        help="Output filename for the generated image"
    )
    
    args = parser.parse_args()
    
    success = generate_image(args.prompt, args.out)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()