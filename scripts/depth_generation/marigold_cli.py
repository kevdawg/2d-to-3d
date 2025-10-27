#!/usr/bin/env python3
"""
marigold_cli.py
Simplified CLI to run Marigold depth estimation pipeline.
Input: Image (should already be preprocessed if needed)
Output: 16-bit depth map PNG
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import sys
import os
import platform

try:
    from diffusers import MarigoldDepthPipeline
except Exception as e:
    print(f"ERROR: could not import required libraries. Ensure you're running this inside the marigold conda env.\n{e}")
    sys.exit(2)

# Windows-safe symbols
def is_windows_cmd():
    if platform.system() != 'Windows':
        return False
    if 'WT_SESSION' in os.environ:  # Windows Terminal
        return False
    return True

if is_windows_cmd():
    OK = "[OK]"
    ERR = "[X]"
    INFO = "[i]"
else:
    OK = "✅"
    ERR = "❌"
    INFO = "ℹ️"


def download_model_if_needed(checkpoint_path: str):
    """
    Check if model exists locally. If not, download it automatically.
    """
    checkpoint = Path(checkpoint_path)
    
    if checkpoint.exists() and (checkpoint / "model_index.json").exists():
        print(f"{INFO} Model found at: {checkpoint}")
        return True
    
    print(f"\n{INFO} Marigold model not found at: {checkpoint}")
    print(f"{INFO} Downloading model for first-time setup...")
    print(f"{INFO} This is a one-time download (~2GB) and may take several minutes.\n")
    
    try:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        
        print("Downloading from HuggingFace: prs-eth/marigold-v1-0")
        pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-v1-0")
        
        print(f"Saving model to: {checkpoint}")
        pipe.save_pretrained(checkpoint)
        
        print(f"\n{OK} Model downloaded and saved successfully!")
        print(f"{OK} Future runs will use the local model (no download needed).\n")
        return True
        
    except Exception as e:
        print(f"\n{ERR} Failed to download model: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check your internet connection")
        print(f"  2. Verify you have ~2GB of free disk space")
        print(f"  3. Try again - downloads can be interrupted")
        print(f"  4. Manual download: https://huggingface.co/prs-eth/marigold-v1-0\n")
        return False


def load_pipe(checkpoint: str):
    """
    Try to load a MarigoldDepthPipeline from a local path.
    """
    if not os.path.isdir(checkpoint):
        print(f"ERROR: Checkpoint path '{checkpoint}' is not a valid directory.")
        sys.exit(1)
    
    print(f"Loading model from: '{checkpoint}'...")
    try:
        pipe = MarigoldDepthPipeline.from_pretrained(checkpoint, local_files_only=True)
        print(f"{OK} Model loaded successfully.")
        return pipe
    except Exception as e:
        print(f"{ERR} Failed to load model from '{checkpoint}': {e}")
        print(f"{ERR} Model may be corrupted. Try deleting the folder and re-running.")
        sys.exit(1)


def save_16bit(depth_arr, out_path: Path):
    """Saves a numpy array as a 16-bit PNG."""
    min_val, max_val = np.nanmin(depth_arr), np.nanmax(depth_arr)
    if max_val <= min_val:
        normalized = np.zeros_like(depth_arr, dtype=np.uint16)
    else:
        normalized = ((depth_arr - min_val) / (max_val - min_val) * 65535.0).astype(np.uint16)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    img = Image.fromarray(normalized.astype(np.uint16))
    
    if img.mode != 'I':
        img = img.convert('I')
    
    img.save(out_path, format='PNG', bits=16)


def main():
    p = argparse.ArgumentParser(description="Run Marigold depth estimation on an image.")
    p.add_argument("--input", required=True, help="Path to the input image file.")
    p.add_argument("--output", required=True, help="Path to save the output 16-bit depth PNG.")
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_checkpoint = os.path.join(script_dir, '..', '..', 'models', 'marigold_model')
    p.add_argument("--checkpoint", default=default_checkpoint, help="Path to the Marigold model directory.")

    p.add_argument("--steps", type=int, default=10, help="Number of inference steps.")
    p.add_argument("--ensemble", type=int, default=1, help="Ensemble size.")
    p.add_argument("--processing_res", type=int, default=768, help="Internal processing resolution.")
    p.add_argument("--match_input_res", action='store_true', help="Resize output to match input resolution.")
    p.add_argument("--no-match_input_res", action='store_false', dest='match_input_res')
    p.set_defaults(match_input_res=True)
    
    p.add_argument("--save_checkpoints", action='store_true', default=False,
                   help="Save intermediate results after processing")
    p.add_argument("--resume", action='store_true', default=False,
                   help="Resume from checkpoint if available")

    args = p.parse_args()

    checkpoint_path = Path(args.output).parent / f"{Path(args.output).stem}_checkpoint.npz"
    
    if args.resume and checkpoint_path.exists():
        print(f"Found checkpoint, resuming from: {checkpoint_path.name}")
        try:
            checkpoint_data = np.load(checkpoint_path)
            depth_array = checkpoint_data['depth_map']
            print(f"Loaded checkpoint depth map: {depth_array.shape[0]}x{depth_array.shape[1]}")
            
            save_path = Path(args.output)
            save_16bit(depth_array, save_path)
            
            if save_path.exists():
                print(f"{OK} Saved depth map: {save_path.name}")
                checkpoint_path.unlink()
                print(f"Checkpoint cleaned up")
            else:
                print(f"{ERR} ERROR: Failed to save depth map to: {save_path}")
                sys.exit(5)
            
            return
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh processing...")

    img = Image.open(args.input).convert("RGB")
    print(f"Loaded image: {img.size[0]}x{img.size[1]}")

    print("Loading Marigold pipeline ...")
    pipe = load_pipe(args.checkpoint)
    
    from diffusers.utils import logging
    logging.set_verbosity_error()
    
    pipe.set_progress_bar_config(
        leave=False,
        dynamic_ncols=True,
        position=0,
        disable=False
    )

    print(f"Running Marigold depth estimation")
    print(f"Steps: {args.steps} | Ensemble: {args.ensemble} | Resolution: {args.processing_res} | Match Input: {args.match_input_res}")

    run_kwargs = {
        "num_inference_steps": args.steps,
        "ensemble_size": args.ensemble,
        "processing_resolution": args.processing_res,
        "match_input_resolution": args.match_input_res,
    }

    try:
        out = pipe(img, **run_kwargs)
    except Exception as e:
        print(f"\n[ERROR] An error occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        
        if args.save_checkpoints:
            print("\nAttempting to save partial results...")
        
        sys.exit(4)

    try:
        prediction = out.prediction
        if isinstance(prediction, list):
            depth_array = np.array(prediction[0])
        else:
            depth_array = np.array(prediction)
        
        depth_array = np.squeeze(depth_array)
        
        print(f"Depth map generated: {depth_array.shape[0]}x{depth_array.shape[1]}")
        
    except AttributeError as e:
        print(f"\n[ERROR] Could not access 'prediction' attribute from pipeline output: {e}")
        print(f"       Output type received: {type(out)}")
        print(f"       Available attributes: {dir(out)}")
        sys.exit(3)

    save_path = Path(args.output)
    save_16bit(depth_array, save_path)
    
    if args.save_checkpoints:
        np.savez_compressed(
            checkpoint_path,
            depth_map=depth_array,
            input_file=str(args.input),
            timestamp=np.datetime64('now')
        )
        print(f"Checkpoint saved: {checkpoint_path.name}")
    
    if save_path.exists():
        print(f"{OK} Saved depth map: {save_path.name}")
        
        if checkpoint_path.exists() and not args.save_checkpoints:
            checkpoint_path.unlink()
    else:
        print(f"{ERR} ERROR: Failed to save depth map to: {save_path}")
        sys.exit(5)


if __name__ == "__main__":
    main()