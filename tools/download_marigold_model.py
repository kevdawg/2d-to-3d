import os
from diffusers import MarigoldDepthPipeline

# Define the destination directory for the model
# This will create a 'marigold_model' folder next to your 'pipeline' and 'marigold' folders.
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, '..', 'marigold_model')

print(f"Downloading Marigold model to: {model_path}")
print("This may take a few minutes and requires an internet connection...")

try:
    # Download and save the model to the specified path
    MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-v1-0").save_pretrained(model_path)
    print("\n✅ Model downloaded and saved successfully!")
    print(f"You can now point your pipeline to the local path: {model_path}")
except Exception as e:
    print(f"\n❌ An error occurred during download: {e}")
    print("Please check your internet connection and try again.")