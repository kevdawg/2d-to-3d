#!/usr/bin/env python3
"""
Marigold Tiled CLI - High Quality Tiled Depth Estimation

This script implements a two-stage tiled inference pipeline to generate
high-resolution depth maps with high-frequency details.

STAGE 1: Global Context Pass
  - Runs Marigold on the entire image at a low resolution (e.g., 768px)
  - Result: A 'global_depth_map.png' which has the correct overall
    shape and depth relationships, but lacks fine detail.

STAGE 2: High-Detail Tiled Pass
  - Chops the original, high-res image into overlapping tiles.
  - Runs Marigold on *each tile* at a high resolution (e.g., 1024px).
  - Result: A folder of 'tile_N_depth.png' files, each with
    incredible detail but poor global context.

STAGE 3: Smart Stitch
  - Creates a new, empty, high-resolution 16-bit depth map.
  - Loops through each high-detail tile and:
    1. Normalizes its depth range to match the global map's range
       for that same area.
    2. Blends it onto the canvas using a feathered (weighted) mask
       to create seamless transitions in the overlapping regions.
  - Result: A single, high-resolution, 16-bit depth map with
    both accurate global depth and sharp local details.
"""

import argparse
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import sys
import os
import platform
import subprocess
import shutil
import time
import logging

# --- TQDM Progress Bar ---
# This is a lightweight, dependency-free progress bar
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create a terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r    {prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def get_conda_env_python(conda_exe_path: str, env_name: str) -> str:
    """
    Finds the absolute path to the python.exe in a specific conda env.
    This allows us to bypass `conda run` and its quoting bugs.
    """
    conda_root = Path(conda_exe_path).parent.parent
    
    possible_paths = [
        conda_root / "envs" / env_name / "python.exe", # Windows
        conda_root / "envs" / env_name / "bin" / "python" # Linux/macOS
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
            
    which_cmd = "where" if platform.system() == "Windows" else "which"
    try:
        result = subprocess.run(
            [conda_exe_path, "run", "-n", env_name, which_cmd, "python"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        if result.returncode == 0 and result.stdout.strip():
            found_path = result.stdout.strip().split('\n')[0]
            if Path(found_path).exists():
                return found_path
    except Exception:
        pass 

    raise FileNotFoundError(f"Could not find python.exe for conda env '{env_name}'. Looked in: {possible_paths[0]}")

def run_cmd(cmd_list, conda_exe, conda_env, log):
    """
    Run a subprocess command using the direct python.exe path.
    This bypasses all shell quoting bugs.
    """
    
    try:
        python_exe = get_conda_env_python(conda_exe, conda_env)
        
        final_cmd = [python_exe] + cmd_list[1:]
        
        # Log the command we are about to run
        log.debug(f"Running subprocess: {' '.join(final_cmd)}")
        
        result = subprocess.run(
            final_cmd,
            stdout=subprocess.PIPE, # Capture stdout
            stderr=subprocess.PIPE,
            text=True, 
            encoding='utf-8', 
            errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        
        # Log the output from the subprocess
        if result.stdout:
            log.debug(f"Subprocess stdout: {result.stdout.strip()}")
        
        if result.returncode != 0:
            log.error(f"Subprocess stderr: {result.stderr.strip()}")
            raise RuntimeError(f"Command failed with exit code {result.returncode}:\n{result.stderr}")
        
        return True
    
    except Exception as e:
        log.error(f"Subprocess failed: {e}\nCommand: {' '.join(final_cmd)}")
        raise RuntimeError(f"Subprocess failed: {e}\nCommand: {' '.join(final_cmd)}")
    

def save_16bit(depth_arr, out_path: Path):
    # ... (this function is unchanged) ...
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


def calculate_tiles(img_width, img_height, tile_size, overlap):
    # ... (this function is unchanged) ...
    tiles = []
    stride = tile_size - overlap
    
    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            x1 = x
            y1 = y
            x2 = min(x + tile_size, img_width)
            y2 = min(y + tile_size, img_height)
            
            if (x2 - x1) < overlap:
                x1 = max(0, img_width - tile_size)
                x2 = img_width
            if (y2 - y1) < overlap:
                y1 = max(0, img_height - tile_size)
                y2 = img_height
                
            tiles.append((x1, y1, x2, y2))
            
            if x2 == img_width:
                break
        if y2 == img_height:
            break
            
    return tiles


def generate_feathered_weights(tile_size, overlap):
    # ... (this function is unchanged) ...
    weights = np.ones((tile_size, tile_size), dtype=np.float32)
    stride = tile_size - overlap
    
    fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    
    weights[:overlap, :] *= fade[:, np.newaxis]
    weights[-overlap:, :] *= fade[::-1, np.newaxis]
    weights[:, :overlap] *= fade[np.newaxis, :]
    weights[:, -overlap:] *= fade[np.newaxis, ::-1]
    
    return weights


def normalize_tile_to_global(tile_depth, global_section):
    # ... (this function is unchanged) ...
    global_min = np.min(global_section)
    global_max = np.max(global_section)
    global_range = global_max - global_min
    
    if global_range == 0:
        return np.full_like(tile_depth, global_min)
        
    tile_min = np.min(tile_depth)
    tile_max = np.max(tile_depth)
    tile_range = tile_max - tile_min
    
    if tile_range == 0:
        return np.full_like(tile_depth, global_min)

    normalized_tile = (tile_depth - tile_min) / tile_range
    rescaled_tile = (normalized_tile * global_range) + global_min
    
    return rescaled_tile


def main():
    parser = argparse.ArgumentParser(description="High-Quality Tiled Marigold Depth Estimation")
    
    # ... (all args are unchanged) ...
    parser.add_argument("--input", required=True, help="Path to the high-resolution input image.")
    parser.add_argument("--output", required=True, help="Path to save the final 16-bit depth PNG.")
    parser.add_argument("--checkpoint", required=True, help="Path to the Marigold model directory.")
    parser.add_argument("--marigold-cli", required=True, help="Path to the standard marigold_cli.py script.")
    parser.add_argument("--global-res", type=int, default=768, help="Processing resolution for the Stage 1 global context pass.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Pixel size for each high-detail tile (e.g., 1024).")
    parser.add_argument("--tile-overlap", type=int, default=256, help="Pixel overlap between tiles (e.g., 256).")
    parser.add_argument("--tile-steps", type=int, default=20, help="Number of inference steps for *each tile*.")
    parser.add_argument("--tile-ensemble", type=int, default=5, help="Ensemble size for *each tile*.")
    parser.add_argument("--conda-exe", required=True, help="Path to the conda executable.")
    parser.add_argument("--conda-env", required=True, help="Name of the marigold conda environment.")
    
    args = parser.parse_args()
    
    start_time_total = time.time()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    work_dir = output_path.parent / f"{output_path.stem}_tiled_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger('tiled_pipeline')
    log.setLevel(logging.DEBUG)

    # Create file handler
    log_file_path = work_dir / "_pipeline.log"
    fh = logging.FileHandler(log_file_path, mode='a') # 'a' for append
    fh.setLevel(logging.INFO)
    fh_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_format)
    log.addHandler(fh)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # Only show INFO by default
    ch_format = logging.Formatter('%(message)s') # Just the message
    ch.setFormatter(ch_format)
    log.addHandler(ch)
    
    print(f"\n--- Tiled Depth Pipeline Initialized ---")
    print(f"  Input: {input_path.name}")
    print(f"  Output: {output_path.name}")
    print(f"  Work Dir: {work_dir.name}")
    print(f"  Config: {args.tile_size}px tiles, {args.tile_overlap}px overlap")
    print("------------------------------------------")

    try:
        # --- STAGE 1: Global Context Pass ---
        print(f"\n[STAGE 1/3] Generating Global Context Map ({args.global_res}px)...")
        start_time_stage1 = time.time()
        
        global_depth_path = work_dir / "global_depth_map.png"
        
        if global_depth_path.exists():
            log.info("  SKIPPING: global_depth_map.png already exists.")
        else:
            cmd_global = [
                "python", "-u", str(args.marigold_cli),
                "--input", str(input_path),
                "--output", str(global_depth_path),
                "--checkpoint", str(args.checkpoint),
                "--steps", "10", 
                "--ensemble", "1",
                "--processing_res", str(args.global_res),
                "--match_input_res" 
            ]
            
            run_cmd(cmd_global, args.conda_exe, args.conda_env, log)
            log.info(f"  Done in {time.time() - start_time_stage1:.1f}s")
        
        global_depth_img = Image.open(global_depth_path)
        global_depth_arr = np.array(global_depth_img).astype(np.float32)
        
        print(f"  Done in {time.time() - start_time_stage1:.1f}s")
        
        # --- STAGE 2: High-Detail Tiled Pass ---
        print(f"\n[STAGE 2/3] Generating High-Detail Tiles...")
        start_time_stage2 = time.time()
        
        img = Image.open(input_path)
        img_width, img_height = img.size
        
        tiles = calculate_tiles(img_width, img_height, args.tile_size, args.tile_overlap)
        tile_work_dir = work_dir / "tiles"
        tile_work_dir.mkdir()
        
        print(f"  Calculated {len(tiles)} tiles.")
        print_progress_bar(0, len(tiles), prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        tile_depth_paths = {}
        for i, (x1, y1, x2, y2) in enumerate(tiles):
            tile_input_path = tile_work_dir / f"tile_{i}_input.png"
            tile_output_path = tile_work_dir / f"tile_{i}_depth.png"
            
            if tile_output_path.exists():
                tile_depth_paths[(x1, y1, x2, y2)] = tile_output_path
                skipped_count += 1
                print_progress_bar(i + 1, len(tiles), prefix = f'Tile {i+1}/{len(tiles)}', suffix = f'(Skipped {skipped_count})', length = 50)
                continue

            # Crop the original image
            tile_img = img.crop((x1, y1, x2, y2))
            tile_img.save(tile_input_path)
            
            cmd_tile = [
                "python", "-u", str(args.marigold_cli),
                "--input", str(tile_input_path),
                "--output", str(tile_output_path),
                "--checkpoint", str(args.checkpoint),
                "--steps", str(args.tile_steps),
                "--ensemble", str(args.tile_ensemble),
                "--processing_res", str(args.tile_size),
                "--match_input_res" 
            ]
            run_cmd(cmd_tile, args.conda_exe, args.conda_env, log)
            
            tile_depth_paths[(x1, y1, x2, y2)] = tile_output_path
            print_progress_bar(i + 1, len(tiles), prefix = f'Tile {i+1}/{len(tiles)}', suffix = 'Complete', length = 50)

        print(f"  Done in {time.time() - start_time_stage2:.1f}s")

        # --- STAGE 3: Smart Stitch ---
        print(f"\n[STAGE 3/3] Stitching {len(tiles)} Tiles...")
        start_time_stage3 = time.time()
        
        final_depth_arr = np.zeros((img_height, img_width), dtype=np.float32)
        weight_sum_arr = np.zeros((img_height, img_width), dtype=np.float32)
        
        feather_weights = generate_feathered_weights(args.tile_size, args.tile_overlap)
        
        print_progress_bar(0, len(tiles), prefix = 'Stitching:', suffix = 'Complete', length = 50)
        
        for i, (x1, y1, x2, y2) in enumerate(tiles):
            tile_w = x2 - x1
            tile_h = y2 - y1
            
            tile_path = tile_depth_paths[(x1, y1, x2, y2)]
            tile_arr = np.array(Image.open(tile_path)).astype(np.float32)
            
            global_section = global_depth_arr[y1:y2, x1:x2]
            
            normalized_tile = normalize_tile_to_global(tile_arr, global_section)
            
            if tile_w == args.tile_size and tile_h == args.tile_size:
                weights = feather_weights
            else:
                weights = feather_weights[:tile_h, :tile_w]
            
            final_depth_arr[y1:y2, x1:x2] += (normalized_tile * weights)
            weight_sum_arr[y1:y2, x1:x2] += weights
            
            print_progress_bar(i + 1, len(tiles), prefix = f'Stitch {i+1}/{len(tiles)}', suffix = 'Complete', length = 50)

        weight_sum_arr[weight_sum_arr == 0] = 1.0
        
        final_depth_arr = final_depth_arr / weight_sum_arr
        
        save_16bit(final_depth_arr, output_path)
        
        print(f"  Done in {time.time() - start_time_stage3:.1f}s")
        
        # Don't delete the whole work dir, just the 'tiles' sub-dir
        if tile_work_dir.exists():
            log.info(f"\nCleaning up 'tiles' directory...")
            shutil.rmtree(tile_work_dir)
        
        print("\n--- Tiled Pipeline Complete! ---")
        print(f"  Final High-Detail Map: {output_path.name}")
        print(f"  Total Time: {time.time() - start_time_total:.1f}s")

    except Exception as e:
        print(f"\n[ERROR] Tiled pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()