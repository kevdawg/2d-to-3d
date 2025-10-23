#!/usr/bin/env python3
"""
Interactive launcher for Marigold -> Extrude pipeline (conda-run based).
- Double-click run_pipeline.bat to start.
- Uses conda run to execute marigold_cli.py and extrude.py in their respective conda envs.
- Uses Gemini for FREE image generation.
"""


import os
import sys
from pathlib import Path
import time
import shutil
import subprocess
import traceback
import threading
from pathlib import Path
import yaml
import platform
from rembg import remove
from PIL import Image
import trimesh
import numpy as np

# Add scripts directory to Python path
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
SCRIPTS_DIR = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR / "depth_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "image_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "model_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "photo_preprocessing"))

# Regional processing imports
from roi_detector import ROIDetector
from region_processor import RegionProcessor
from depth_blender import DepthBlender

# Import background removal functions
from background_remover_removebg import remove_background
from depth_masking import mask_depth_with_alpha


if not CONFIG_PATH.exists():
    print("Missing config.yaml in pipeline folder. Create it from the example.")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Config values with defaults
CONDA_EXE = cfg.get("conda_exe", "conda")

# --- FIX: Auto-detect full conda path to improve compatibility ---
if CONDA_EXE == "conda":
    conda_path = shutil.which("conda")
    if conda_path:
        print(f"Auto-detected conda executable at: {conda_path}")
        CONDA_EXE = conda_path
    else:
        print("\nWARNING: Could not find 'conda' in the system PATH.")
        print("         Please specify the full path to 'conda.exe' or 'conda.bat' in config.yaml if you encounter errors.")

# Environment names
AIGEN_ENV = cfg.get("aigen_env", "aigen")
MARIGOLD_ENV = cfg.get("marigold_env", "marigold")
DEPTH_ENV = cfg.get("depth_env", "depth-to-3d")
PHOTO_PREP_ENV = cfg.get("photo_prep_env", "photo-prep")
USE_CONDA = bool(cfg.get("use_conda", True))

# Script paths (relative to HERE which is pipeline/)
MARIGOLD_CLI = (HERE / cfg.get("marigold_cli", "../scripts/depth_generation/marigold_cli.py")).resolve()
EXTRUDE_CLI = (HERE / cfg.get("extrude_cli", "../scripts/model_generation/extrude_cli.py")).resolve()

# Directory paths - FIXED (relative to HERE which is pipeline/)
DIR_AI_GENERATED = (HERE / cfg.get("dir_ai_generated", "../data/AI_files")).resolve()
DIR_PHOTOS = (HERE / cfg.get("dir_photos", "../data/Photos")).resolve()
DIR_3D = (HERE / cfg.get("dir_3d", "../data/3D_files")).resolve()
DIR_ENHANCED = (HERE / cfg.get("dir_enhanced", "../data/Photos_enhanced")).resolve()

# Background removal settings
REMOVE_BACKGROUND = bool(cfg.get("remove_background", True))
BG_REMOVAL_METHOD = cfg.get("bg_removal_method", "rembg")
BG_REMOVAL_MODEL = cfg.get("bg_removal_model", "isnet-general-use")
BG_CROP_ENABLED = bool(cfg.get("bg_crop_enabled", True))
BG_CROP_MARGIN = int(cfg.get("bg_crop_margin", 10))
REMOVEBG_API_KEY = os.environ.get('REMOVEBG_API_KEY')

# Load presets and defaults from config
MARIGOLD_PRESETS = cfg.get("marigold_presets", {})
EXTRUDE_DEFAULTS = cfg.get("extrude_defaults", {})

# Ensure folders exist - FIXED (no relative_to() call)
print("Initializing directories...")
for d in (DIR_AI_GENERATED, DIR_PHOTOS, DIR_3D, DIR_ENHANCED):
    d.mkdir(parents=True, exist_ok=True)
    # Simple display without relative_to() to avoid path errors
    print(f"  ✓ {d.name}/ -> {d}")

# Windows-safe symbols
def is_windows_cmd():
    if platform.system() != 'Windows':
        return False
    if 'WT_SESSION' in os.environ:
        return False
    return True

if is_windows_cmd():
    OK = "[OK]"
    ERR = "[X]"
    WARN = "[!]"
    TRASH = "[DEL]"
    INFO = "[i]"
else:
    OK = "✅"
    ERR = "❌"
    WARN = "⚠️"
    TRASH = "🗑️"
    INFO = "ℹ️"


def run_cmd(cmd_list, show_timer=False, timer_message="Processing"):
    """
    Run a subprocess command with clean single-line progress bar display.
    """
    import time
    
    output_lines = []
    
    try:
        proc = subprocess.Popen(
            cmd_list, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace', 
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            bufsize=1  # Line buffered
        )
        
        start_time = time.time()
        last_progress_line = None
        
        # Process output line-by-line
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
                
            line = line.rstrip('\n\r')
            output_lines.append(line)
            
            # Check if this is a progress bar line (contains % or it/s)
            is_progress = '%|' in line or 'it/s' in line or 'it]' in line
            
            if is_progress:
                # Progress bar - update on same line
                # Clear previous line and write new one
                sys.stdout.write('\r' + ' ' * 100 + '\r')  # Clear line
                sys.stdout.write('    ' + line)
                sys.stdout.flush()
                last_progress_line = line
            else:
                # Regular output - print on new line
                if last_progress_line:
                    # If we just had progress bars, move to new line first
                    sys.stdout.write('\n')
                    last_progress_line = None
                sys.stdout.write('    ' + line + '\n')
                sys.stdout.flush()
        
        proc.wait()
        
        # Clear any remaining progress bar
        if last_progress_line:
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.flush()

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        if proc.returncode == 0:
            print(f"  {OK} Completed successfully (completed in {time_str})")
        else:
            print(f"  {ERR} Failed with exit code {proc.returncode} (completed in {time_str})")

        return proc.returncode, "\n".join(output_lines)
    
    except FileNotFoundError:
        err_msg = f"Error: Command '{cmd_list[0]}' not found."
        print(f"  {ERR} {err_msg}")
        raise RuntimeError(err_msg)
    except Exception as e:
        err_msg = f"An unexpected error occurred: {e}"
        print(f"  {ERR} {err_msg}")
        raise RuntimeError(err_msg)


def conda_prefix_cmd(env_name, cmd_list):
    """Return a full command list that runs cmd_list inside conda env."""
    return [CONDA_EXE, "run", "-n", env_name, "--no-capture-output"] + cmd_list if USE_CONDA else cmd_list


def get_next_folder_name(base_name: str, parent_dir: Path) -> str:
    """
    Generate a unique folder name by appending a number if needed.
    E.g., "oak_leaf", "oak_leaf_2", "oak_leaf_3"
    """
    # Sanitize the base name
    safe = "".join([c if c.isalnum() or c in ("-", "_", " ") else "_" for c in base_name])
    safe = safe.strip().replace(" ", "_")[:50]
    
    # Check if folder exists, increment number if needed
    folder_name = safe
    counter = 2
    while (parent_dir / folder_name).exists():
        folder_name = f"{safe}_{counter}"
        counter += 1
    
    return folder_name


def safe_name_from_prompt(prompt: str) -> str:
    """
    Create a safe folder name from the AI prompt description.
    """
    return get_next_folder_name(prompt, DIR_3D)


def safe_name_from_file(file_path: Path) -> str:
    """
    Create a safe folder name from an existing file.
    Uses the base filename, removing timestamps and quality suffixes.
    """
    base = file_path.stem
    
    # Remove any existing timestamp patterns like _20251008_232149_575c44
    import re
    base = re.sub(r'_\d{8}_\d{6}_[a-f0-9]{6}', '', base)
    
    # Remove quality suffix if present (avoid "frog_low_quality_low_quality")
    for quality in ['_low_quality', '_medium_quality', '_high_quality']:
        if base.endswith(quality):
            base = base[:-len(quality)]
    
    # Use get_next_folder_name to ensure uniqueness
    return get_next_folder_name(base, DIR_3D)

def remove_background_if_enabled(image_path: Path, output_path: Path = None) -> Path:
    """
    Remove background from image if enabled in config.
    Uses rembg library for background removal.
    Returns path to processed image (either cleaned or original).
    """
    if not REMOVE_BACKGROUND:
        return image_path
    
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_nobg.png"
    
    try:
        print(f"  Removing background from {image_path.name}...")
        with Image.open(image_path) as input_img:
            # Remove background using rembg
            output_img = remove(input_img)
            
            # Auto-crop transparent borders if enabled
            if BG_CROP_ENABLED:
                # Get bounding box of non-transparent pixels
                bbox = output_img.getbbox()
                if bbox:
                    # Add margin
                    width, height = output_img.size
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, x1 - BG_CROP_MARGIN)
                    y1 = max(0, y1 - BG_CROP_MARGIN)
                    x2 = min(width, x2 + BG_CROP_MARGIN)
                    y2 = min(height, y2 + BG_CROP_MARGIN)
                    
                    # Crop
                    output_img = output_img.crop((x1, y1, x2, y2))
                    print(f"    Cropped: {width}x{height} → {output_img.width}x{output_img.height}")
            
            output_img.save(output_path, 'PNG')
        
        print(f"  {OK} Background removed (transparent)")
        return output_path
        
    except Exception as e:
        print(f"  {WARN} Background removal failed: {e}")
        print(f"  {WARN} Continuing with original image...")
        return image_path


def generate_via_gemini(user_desc: str, filename_out: Path):
    """Call generate_with_gemini.py helper via subprocess in NO environment (uses base Python)."""
    gen_py = SCRIPTS_DIR / "image_generation" / "generate_with_gemini.py"
    if not gen_py.exists():
        raise RuntimeError(f"generate_with_gemini.py not found at {gen_py}")

    # Run in gemini conda environment
    cmd = ["python", str(gen_py), "--prompt", user_desc, "--out", str(filename_out)]
    full_cmd = conda_prefix_cmd(AIGEN_ENV, cmd)
    
    rc, output = run_cmd(full_cmd)
    if rc != 0:
        raise RuntimeError(f"Image generation failed.\n\nOutput:\n{output}")
    return filename_out



def generate_via_imagen3(user_desc: str, filename_out: Path):
    """Call generate_with_imagen3.py helper via subprocess in aigen conda environment."""
    gen_py = SCRIPTS_DIR / "image_generation" / "generate_with_imagen3.py"
    if not gen_py.exists():
        raise RuntimeError(f"generate_with_imagen3.py not found at {gen_py}")
    
    # Get aigen environment name from config
    aigen_env = cfg.get("aigen_env", "aigen")
    
    # Run in aigen conda environment with suppressed stderr
    cmd = ["python", str(gen_py), "--prompt", user_desc, "--out", str(filename_out)]
    full_cmd = conda_prefix_cmd(aigen_env, cmd)
    
    # Run with custom error filtering
    import subprocess
    try:
        # Set environment variables to suppress gRPC warnings
        env = os.environ.copy()
        env['GRPC_VERBOSITY'] = 'ERROR'
        env['GLOG_minloglevel'] = '2'
        
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        stdout, stderr = proc.communicate()
        
        # Print stdout
        if stdout:
            for line in stdout.strip().split('\n'):
                if line.strip():
                    print(f"    {line}")
        
        if proc.returncode != 0:
            raise RuntimeError(f"Image generation failed with exit code {proc.returncode}")
        
        return filename_out
        
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")


# ============================================
# INTERACTIVE WRAPPERS
# ============================================

def generate_ai_image_menu():
    """Submenu for AI image generation."""
    
    print(f"\n{'─'*60}")
    print("GENERATE AI IMAGE")
    print('─'*60)
    print("  1. Gemini (FREE, basic quality)")
    print("  2. Imagen ($0.04, high quality)")
    print("  3. Back to main menu")
    print('─'*60)
    
    choice = input("\nSelect option [1-3]: ").strip()
    
    if choice == "1":
        generate_with_gemini_interactive()
    elif choice == "2":
        generate_with_aigen_interactive()
    elif choice == "3":
        return
    else:
        print(f"\n{ERR} Invalid option.")
        generate_ai_image_menu()


def generate_with_gemini_interactive():
    """Interactive Gemini image generation."""
    
    print(f"\n{'─'*60}")
    print("GENERATE WITH GEMINI (FREE)")
    print('─'*60)
    
    prompt = input("\nEnter image description (or 'cancel'): ").strip()
    if prompt.lower() == "cancel":
        return
    
    # Create safe filename from prompt
    safe_prompt = "".join([c if c.isalnum() or c in ("-", "_", " ") else "_" for c in prompt])
    safe_prompt = safe_prompt.strip().replace(" ", "_")[:50]
    
    # Check if output file already exists
    out_path = DIR_AI_GENERATED / f"{safe_prompt}.png"
    counter = 2
    while out_path.exists():
        out_path = DIR_AI_GENERATED / f"{safe_prompt}_{counter}.png"
        counter += 1
    
    try:
        start_time = time.time()
        
        print(f"\nGenerating with Gemini (FREE tier)...")
        generate_via_gemini(prompt, out_path)
        
        # Show timing
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        print(f"\n{OK} Image saved: {out_path.name}")
        print(f"   Generation time: {time_str}")
        print(f"   Saved to: AI_files/{out_path.name}")
        
    except Exception as e:
        print(f"\n{ERR} Image generation failed: {e}")
    
    input("\nPress Enter to continue...")


def generate_with_aigen_interactive():
    """Interactive Imagen 3 image generation."""
    
    print(f"\n{'─'*60}")
    print("GENERATE WITH AIGEN 3 ($0.04)")
    print('─'*60)
    
    prompt = input("\nEnter image description (or 'cancel'): ").strip()
    if prompt.lower() == "cancel":
        return
    
    # Create safe filename from prompt
    safe_prompt = "".join([c if c.isalnum() or c in ("-", "_", " ") else "_" for c in prompt])
    safe_prompt = safe_prompt.strip().replace(" ", "_")[:50]
    
    # Check if output file already exists
    out_path = DIR_AI_GENERATED / f"{safe_prompt}.png"
    counter = 2
    while out_path.exists():
        out_path = DIR_AI_GENERATED / f"{safe_prompt}_{counter}.png"
        counter += 1
    
    try:
        start_time = time.time()
        
        print(f"\nGenerating with Imagen 3 (high quality)...")
        generate_via_imagen3(prompt, out_path)
        
        # Show timing
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        print(f"\n{OK} Image saved: {out_path.name}")
        print(f"   Generation time: {time_str}")
        print(f"   Cost: ~$0.04")
        print(f"   Saved to: AI_files/{out_path.name}")
        
    except Exception as e:
        print(f"\n{ERR} Image generation failed: {e}")
    
    input("\nPress Enter to continue...")
        

def run_marigold_cli(image_path: Path, depth_out: Path, marigold_opts: dict):
    """Run marigold_cli.py to create a 16-bit depth PNG."""
    marigold_model_path = HERE / ".." / "models" / "marigold_model"
    if not marigold_model_path.exists():
        raise RuntimeError(f"Marigold model not found at {marigold_model_path}.")

    cmd = ["python", str(MARIGOLD_CLI),
           "--input", str(image_path),
           "--output", str(depth_out),
           "--checkpoint", str(marigold_model_path),
           "--steps", str(marigold_opts.get("marigold_steps")),
           "--ensemble", str(marigold_opts.get("marigold_ensemble")),
           "--processing_res", str(marigold_opts.get("marigold_processing_res"))]
    
    if marigold_opts.get("marigold_match_input_res"):
        cmd.append("--match_input_res")
    else:
        cmd.append("--no-match_input_res")
    
    if marigold_opts.get("marigold_save_checkpoints", False):
        cmd.append("--save_checkpoints")
    
    if marigold_opts.get("marigold_resume", False):
        cmd.append("--resume")
    
    # ADD THIS: Show exact command for manual testing
    print(f"\n💻 Marigold command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    full = conda_prefix_cmd(MARIGOLD_ENV, cmd)
    
    print(f"\nGenerating depth map from {image_path.name}...")
    rc, output = run_cmd(full)
    
    if rc != 0:
        last_lines = "\n".join(output.splitlines()[-5:])
        raise RuntimeError(f"Marigold depth generation failed.\n\nLast output from script:\n{last_lines}")
    return depth_out


def run_marigold_with_regions(image_path: Path, depth_out: Path, config: dict):
    """
    Run Marigold with region-specific processing for faces/subjects vs background.
    Includes automatic background removal before detection.
    
    Args:
        image_path: Path to input image
        depth_out: Path for final depth map output
        config: Full config dict from config.yaml
    
    Returns:
        Path to final depth map
    """
    region_config = config.get('region_processing', {})
    
    # Check if regional processing is enabled
    if not region_config.get('enabled', False):
        # Fall back to standard processing
        marigold_opts = config['marigold_presets']['high_quality']
        return run_marigold_cli(image_path, depth_out, marigold_opts)
    
    print(f"\n{'='*60}")
    print(f"  Regional Depth Processing")
    print(f"{'='*60}")
    
    # Create working directory for intermediate files
    work_dir = depth_out.parent / f"{depth_out.stem}_regions"
    work_dir.mkdir(exist_ok=True)
    
    # STEP 0: Remove background if enabled (should already be done, but double-check)
    working_image = image_path
    if config.get('remove_background', False) and not str(image_path).endswith('_nobg.png'):
        print("\n[0/5] Background removal (if not already done)...")
        nobg_path = image_path.parent / f"{image_path.stem}_nobg.png"
        if nobg_path.exists():
            print(f"   ✓ Using existing: {nobg_path.name}")
            working_image = nobg_path
        else:
            # Background should have been removed earlier, but do it now if missing
            working_image = remove_background_if_enabled(image_path, nobg_path)
    
    # STEP 1: Detect regions of interest
    print("\n[1/5] Detecting subjects...")
    detection_mode = region_config.get('detection_mode', 'human')
    print(f"   Detection mode: {detection_mode}")
    
    detector = ROIDetector(
        use_sam=region_config.get('use_sam', True),
        detection_mode=detection_mode
    )

    # Add prompt settings if using prompt mode
    if detection_mode == 'prompt':
        detector.detection_prompt = region_config.get('detection_prompt', 'animal face')
        detector.prompt_confidence = region_config.get('prompt_confidence', 0.25)
    
    regions = detector.create_region_masks(
        str(working_image),
        detection_mode=detection_mode
    )
    
    num_subjects = len(regions['faces'])
    print(f"   Found {num_subjects} subject(s)")
    
    # Save visualization
    vis_path = work_dir / "detected_regions.jpg"
    detector.visualize_regions(working_image, regions, vis_path)
    
    # STEP 2: Process each region with custom settings
    print("\n[2/4] Processing regions with custom settings...")
    
    processor = RegionProcessor(
        marigold_cli_path=MARIGOLD_CLI,
        conda_exe=CONDA_EXE,
        marigold_env=MARIGOLD_ENV
    )
    
    depth_maps = []
    masks = []
    
    # Process faces
    for i, (face_mask, face_box) in enumerate(zip(regions['faces'], regions['face_boxes'])):
        region_name = f"face_{i+1}"
        face_settings = region_config['face']
        
        depth = processor.process_region(
            image_path=working_image,
            mask=face_mask,
            region_name=region_name,
            preprocess_settings=face_settings['preprocessing'],
            marigold_settings=face_settings['marigold'],
            output_dir=work_dir
        )
        
        depth_maps.append(depth)
        masks.append(face_mask)
        print(f"      ✓ Processed {region_name}")
    
    # Process background
    bg_settings = region_config['background']
    bg_depth = processor.process_region(
        image_path=working_image,
        mask=regions['background'],
        region_name="background",
        preprocess_settings=bg_settings['preprocessing'],
        marigold_settings=bg_settings['marigold'],
        output_dir=work_dir
    )
    
    depth_maps.append(bg_depth)
    masks.append(regions['background'])
    print(f"      ✓ Processed background")
    
    # STEP 3: Blend depth maps
    print("\n[3/4] Blending depth maps...")
    blender = DepthBlender(blend_width=region_config.get('blend_width', 30))
    
    final_depth = blender.blend_depth_maps(
        depth_maps=depth_maps,
        masks=masks,
        method='weighted',
        normalize=True
    )
    
    # Save blend visualization
    blend_vis_path = work_dir / "blend_visualization.jpg"
    blender.visualize_blend(depth_maps, masks, blend_vis_path)
    
    # STEP 4: Save final depth map
    print("\n[4/4] Saving final depth map...")
    Image.fromarray(final_depth, mode='I;16').save(depth_out)
    
    print(f"\n{OK} Regional processing complete!")
    print(f"   Final depth: {depth_out.name}")
    print(f"   Intermediate files: {work_dir.name}/")
    
    return depth_out


def run_extrude_cli(depth_path: Path, stl_out: Path, extrude_params: dict):
    """Call extrude.py to produce STL from depth map."""
    cmd = ["python", str(EXTRUDE_CLI),
           "--input", str(depth_path),
           "--output", str(stl_out),
           "--width_mm", str(extrude_params.get("width_mm", 100.0)),
           "--smoothing", str(extrude_params.get("smoothing", 3)),
           "--near_offset", str(extrude_params.get("near_offset", 0.0)),
           "--far_offset", str(extrude_params.get("far_offset", 1.0)),
           "--emboss", str(extrude_params.get("emboss", 0.3)),
           "--f_thic", str(extrude_params.get("f_thic", 0.05)),
           "--f_near", str(extrude_params.get("f_near", -0.15)),
           "--f_back", str(extrude_params.get("f_back", 0.01)),
           "--vertex_colors", str(extrude_params.get("vertex_colors", True)),
           "--scene_lights", str(extrude_params.get("scene_lights", True)),
           "--prepare_for_3d_printing", str(extrude_params.get("prepare_for_3d_printing", False)),
           "--zip_outputs", str(extrude_params.get("zip_outputs", False))]

    # Print extrude_cli.py command to window for the user
    print(f"\n💻 Extrusion command:")
    print(f"   {' '.join(cmd)}")
    print()

    full = conda_prefix_cmd(DEPTH_ENV, cmd)
    
    print(f"\nConverting depth map to 3D model...")
    rc, output = run_cmd(full)
    
    if rc != 0:
        last_lines = "\n".join(output.splitlines()[-5:])
        raise RuntimeError(f"3D extrusion failed.\n\nLast output from script:\n{last_lines}")
    return stl_out
    

def process_single(image_path: Path, marigold_opts: dict, extrude_opts: dict, file_counter=None, quality_preset="high"):
    """Process a single image end-to-end."""
    try:
        start_time = time.time()
        
        counter_str = f"[{file_counter}] " if file_counter else ""
        print(f"\n{'='*60}\n{counter_str}Processing: {image_path.name}\n{'='*60}")
        
        if REMOVE_BACKGROUND:
            print(f"Background removal: ENABLED ({BG_REMOVAL_METHOD})")
        else:
            print(f"Background removal: DISABLED")
        
        # Create safe folder name with quality suffix
        name = safe_name_from_file(image_path)
        name_with_quality = f"{name}_{quality_preset}"
        run_dir = DIR_3D / name_with_quality
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # STEP 1: Remove background if enabled
        working_image = image_path
        if REMOVE_BACKGROUND:
            nobg_path = run_dir / f"{name_with_quality}_nobg.png"
            working_image = remove_background_if_enabled(image_path, nobg_path)
            # Copy cleaned image as "original" to output folder
            shutil.copy2(working_image, run_dir / f"{name_with_quality}_original.png")
        else:
            # Copy original to output folder
            shutil.copy2(image_path, run_dir / image_path.name)

        # Define output paths
        depth_out = run_dir / f"{name_with_quality}_depth_16bit.png"

        # STEP 2: Prepare image for Marigold (composite transparent images onto neutral background)
        if REMOVE_BACKGROUND:
            prepared_path = run_dir / f"{name_with_quality}_prepared.png"
            
            # Open image
            img = Image.open(working_image)
            
            # Check if has alpha channel
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                print(f"  Compositing onto neutral background for Marigold...")
                
                # Convert to RGBA
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Create gray background (neutral for depth)
                background = Image.new('RGBA', img.size, (128, 128, 128, 255))
                
                # Composite
                composited = Image.alpha_composite(background, img)
                
                # Convert to RGB
                rgb_img = composited.convert('RGB')
                rgb_img.save(prepared_path)
                
                marigold_input = prepared_path
                print(f"  ✓ Prepared: {prepared_path.name}")
            else:
                marigold_input = working_image
        else:
            marigold_input = working_image

        # STEP 3: Run Marigold (with regional processing if enabled)
        if cfg.get('region_processing', {}).get('enabled', False):
            run_marigold_with_regions(marigold_input, depth_out, cfg)
        else:
            run_marigold_cli(marigold_input, depth_out, marigold_opts)

        # STEP 4: Run extrusion
        stl_out = run_dir / f"{name_with_quality}.stl"
        run_extrude_cli(depth_out, stl_out, extrude_opts)
        
        # CALCULATE TOTAL TIME
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        # Verify output files
        print(f"\n{OK} Processing complete! Output folder: {run_dir.name}")
        print(f"   Total processing time ({quality_preset} quality): {time_str}")
        
        output_files = [
            (run_dir / f"{name_with_quality}_original.png" if REMOVE_BACKGROUND else run_dir / image_path.name, "Original"),
            (depth_out, "Depth map"),
            (stl_out, "STL"),
            (run_dir / f"{name_with_quality}.glb", "GLB"),
            (run_dir / f"{name_with_quality}.obj", "OBJ")
        ]
        
        all_exist = True
        for file_path, description in output_files:
            if file_path.exists():
                print(f"   {OK} {description}: {file_path.name}")
            else:
                print(f"   {ERR} {description}: MISSING!")
                all_exist = False
        
        if all_exist:
            # Delete the original file from 2D_files after successful processing
            try:
                image_path.unlink()
                print(f"\n   {TRASH} Deleted original from 2D_files: {image_path.name}")
            except Exception as e:
                print(f"\n   {WARN} Could not delete original: {e}")
        else:
            print(f"\n   {WARN} Some files missing - keeping original in 2D_files")
        
    except Exception as e:
        print(f"\n  [ERROR] Failed to process {image_path.name}: {e}")
        import traceback
        traceback.print_exc()


def select_and_process(quality_preset):
    """
    Select source (AI or photo) and process with specified quality.
    """
    
    print(f"\n{'─'*60}")
    print(f"SELECT IMAGE SOURCE")
    print('─'*60)
    print("  1. From AI_files/ folder")
    print("  2. From Photos/ folder (will auto-enhance)")
    print("  3. Back")
    print('─'*60)
    
    source_choice = input("\nSelect source [1-3]: ").strip()
    
    if source_choice == "1":
        source_dir = DIR_AI_GENERATED
        auto_enhance = False
        print(f"\n📁 Scanning: {source_dir.relative_to(HERE.parent)}")
    elif source_choice == "2":
        source_dir = DIR_PHOTOS
        auto_enhance = cfg.get("auto_enhance_photos", True)
        print(f"\n📁 Scanning: {source_dir.relative_to(HERE.parent)}")
    elif source_choice == "3":
        return
    else:
        print(f"\n{ERR} Invalid option.")
        return
    
    # List available images
    files = list_image_files(source_dir)
    
    if not files:
        print(f"\n{WARN} No images found in {source_dir.name}/")
        print(f"       Path checked: {source_dir}")
        if source_choice == "1":
            print(f"       Generate AI images first (Main Menu → Option 1)")
        else:
            print(f"       Add photos to: {source_dir}")
        input("\nPress Enter to continue...")
        return
    
    print(f"\nAvailable images in {source_dir.name}/:")
    for i, file_path in enumerate(files, 1):
        print(f"  {i}. {file_path.name}")
    
    print(f"  {len(files) + 1}. Back")
    
    try:
        choice = input(f"\nSelect image [1-{len(files) + 1}]: ").strip()
        choice_num = int(choice)
        
        if choice_num == len(files) + 1:
            return
        
        if 1 <= choice_num <= len(files):
            image_path = files[choice_num - 1]
            process_single_image(image_path, quality_preset, auto_enhance)
        else:
            print(f"\n{ERR} Invalid selection.")
    
    except ValueError:
        print(f"\n{ERR} Invalid input.")


def view_edit_defaults():
    print("\nDefault parameters are managed via presets in 'config.yaml'.")
    print("You can edit photo preprocessing, depth generation, and 3D model extrusion settings here.")
    print("'High', 'medium', and 'low' quality presets can be adjusted as desirable.")
    print(f"\nOpening config file: {CONFIG_PATH}\n")
    
    try:
        if sys.platform == "win32":
            # Use os.startfile with no console output
            import subprocess
            subprocess.Popen(
                ['cmd', '/c', 'start', '', str(CONFIG_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        elif sys.platform == "darwin":  # macOS
            subprocess.Popen(
                ["open", str(CONFIG_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:  # linux
            subprocess.Popen(
                ["xdg-open", str(CONFIG_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        print("Config file opened in default editor.")
        
    except Exception as e:
        print(f"Could not open config file automatically. Please open it manually at:\n{CONFIG_PATH}")
    
    input("\nPress Enter to return to the main menu after editing.")


def main_menu():
    """Simplified main menu with logical grouping."""
    
    print(f"\n{'='*60}")
    print(f"  2D to 3D Pipeline - Bas-Relief Generator")
    print(f"{'='*60}")
    
    while True:
        print("\n" + "-"*60)
        print("MAIN MENU:")
        print("-"*60)
        print("  1. Generate AI Image")
        print("  2. Transform 2D to 3D")
        print("  3. Rerun Depth-to-Model (new settings)")
        print("  4. Edit Configuration")
        print("  5. Quit")
        print("-"*60)
        
        choice = input("\nSelect option [1-5]: ").strip()
        
        if choice == "1":
            generate_ai_image_menu()
        elif choice == "2":
            transform_2d_to_3d_menu()
        elif choice == "3":
            rerun_depth_to_model_menu()
        elif choice == "4":
            edit_configuration()
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print(f"\n{ERR} Invalid option. Please choose 1-5.")


def generate_ai_image_menu():
    """Submenu for AI image generation."""
    
    print(f"\n{'─'*60}")
    print("GENERATE AI IMAGE")
    print('─'*60)
    print("  1. Gemini (FREE, basic quality)")
    print("  2. Imagen ($0.04, high quality)")
    print("  3. Back to main menu")
    print('─'*60)
    
    choice = input("\nSelect option [1-3]: ").strip()
    
    if choice == "1":
        generate_with_gemini_interactive()
    elif choice == "2":
        generate_with_aigen_interactive()
    elif choice == "3":
        return
    else:
        print(f"\n{ERR} Invalid option.")
        generate_ai_image_menu()


def generate_with_gemini_interactive():
    """Interactive Gemini image generation."""
    
    print(f"\n{'─'*60}")
    print("GENERATE WITH GEMINI (FREE)")
    print('─'*60)
    
    prompt = input("\nEnter image description (or 'cancel'): ").strip()
    if prompt.lower() == "cancel":
        return
    
    # Create safe filename from prompt
    safe_prompt = "".join([c if c.isalnum() or c in ("-", "_", " ") else "_" for c in prompt])
    safe_prompt = safe_prompt.strip().replace(" ", "_")[:50]
    
    # Check if output file already exists
    out_path = DIR_AI_GENERATED / f"{safe_prompt}.png"
    counter = 2
    while out_path.exists():
        out_path = DIR_AI_GENERATED / f"{safe_prompt}_{counter}.png"
        counter += 1
    
    try:
        start_time = time.time()
        
        print(f"\nGenerating with Gemini (FREE tier)...")
        generate_via_gemini(prompt, out_path)
        
        # Show timing
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        print(f"\n{OK} Image saved: {out_path.name}")
        print(f"   Generation time: {time_str}")
        
    except Exception as e:
        print(f"\n{ERR} Image generation failed: {e}")
    
    input("\nPress Enter to continue...")


def generate_with_aigen_interactive():
    """Interactive Imagen 3 image generation."""
    
    print(f"\n{'─'*60}")
    print("GENERATE WITH AIGEN 3 ($0.04)")
    print('─'*60)
    
    prompt = input("\nEnter image description (or 'cancel'): ").strip()
    if prompt.lower() == "cancel":
        return
    
    # Create safe filename from prompt
    safe_prompt = "".join([c if c.isalnum() or c in ("-", "_", " ") else "_" for c in prompt])
    safe_prompt = safe_prompt.strip().replace(" ", "_")[:50]
    
    # Check if output file already exists
    out_path = DIR_AI_GENERATED / f"{safe_prompt}.png"
    counter = 2
    while out_path.exists():
        out_path = DIR_AI_GENERATED / f"{safe_prompt}_{counter}.png"
        counter += 1
    
    try:
        start_time = time.time()
        
        print(f"\nGenerating with Imagen 3 (high quality)...")
        generate_via_imagen3(prompt, out_path)
        
        # Show timing
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        print(f"\n{OK} Image saved: {out_path.name}")
        print(f"   Generation time: {time_str}")
        print(f"   Cost: ~$0.04")
        
    except Exception as e:
        print(f"\n{ERR} Image generation failed: {e}")
    
    input("\nPress Enter to continue...")


def transform_2d_to_3d_menu():
    """Submenu for 2D to 3D transformation."""
    
    print(f"\n{'─'*60}")
    print("TRANSFORM 2D TO 3D")
    print('─'*60)
    print("  1. Low Quality (fast, ~2 min)")
    print("  2. Medium Quality (balanced, ~5 min)")
    print("  3. High Quality (best, ~10 min)")
    print("  4. Batch Process Folder")
    print("  5. Back to main menu")
    print('─'*60)
    
    choice = input("\nSelect option [1-5]: ").strip()
    
    if choice in ["1", "2", "3"]:
        quality = ["low_quality", "medium_quality", "high_quality"][int(choice) - 1]
        select_and_process(quality)
    elif choice == "4":
        batch_process_folder(quality="high_quality")
    elif choice == "5":
        return
    else:
        print(f"\n{ERR} Invalid option.")
        transform_2d_to_3d_menu()


def rerun_depth_to_model_menu():
    """
    Reprocess existing depth maps with new extrusion settings.
    Useful for fine-tuning without regenerating depth.
    """
    
    print(f"\n{'─'*60}")
    print("RERUN DEPTH-TO-MODEL")
    print('─'*60)
    print("This will reprocess an existing depth map with new settings.")
    print("Useful for adjusting relief height, smoothing, etc.")
    print('─'*60)
    
    # Find all depth maps in 3D_files
    depth_maps = []
    for project_dir in DIR_3D.iterdir():
        if project_dir.is_dir():
            for depth_file in project_dir.glob("*_depth_16bit.png"):
                depth_maps.append(depth_file)
    
    if not depth_maps:
        print(f"\n{WARN} No depth maps found. Generate 3D models first (option 2).")
        input("\nPress Enter to continue...")
        return
    
    print(f"\nFound {len(depth_maps)} depth map(s):")
    for i, depth_path in enumerate(depth_maps, 1):
        project_name = depth_path.parent.name
        print(f"  {i}. {project_name}")
    
    print(f"  {len(depth_maps) + 1}. Back to main menu")
    
    try:
        choice = input(f"\nSelect depth map [1-{len(depth_maps) + 1}]: ").strip()
        choice_num = int(choice)
        
        if choice_num == len(depth_maps) + 1:
            return
        
        if 1 <= choice_num <= len(depth_maps):
            depth_path = depth_maps[choice_num - 1]
            
            # Ask if they want to edit settings first
            print(f"\nCurrent settings in config.yaml will be used.")
            edit = input("Edit settings now? [y/N]: ").strip().lower()
            
            if edit in ['y', 'yes']:
                edit_configuration()
            
            # Rerun extrusion with current config
            reprocess_depth_map(depth_path)
        else:
            print(f"\n{ERR} Invalid selection.")
    
    except ValueError:
        print(f"\n{ERR} Invalid input.")


def select_and_process(quality_preset):
    """
    Select source (AI or photo) and process with specified quality.
    """
    
    print(f"\n{'─'*60}")
    print(f"SELECT IMAGE SOURCE")
    print('─'*60)
    print("  1. From AI_files/ folder")
    print("  2. From photos/ folder (will auto-enhance)")
    print("  3. Back")
    print('─'*60)
    
    source_choice = input("\nSelect source [1-3]: ").strip()
    
    if source_choice == "1":
        source_dir = DIR_AI_GENERATED
        auto_enhance = False
    elif source_choice == "2":
        source_dir = DIR_PHOTOS
        auto_enhance = cfg.get("auto_enhance_photos", True)
    elif source_choice == "3":
        return
    else:
        print(f"\n{ERR} Invalid option.")
        return
    
    # List available images
    files = list_image_files(source_dir)
    
    if not files:
        print(f"\n{WARN} No images found in {source_dir.name}/")
        print(f"       Generate AI images or add photos first.")
        input("\nPress Enter to continue...")
        return
    
    print(f"\nAvailable images in {source_dir.name}/:")
    for i, file_path in enumerate(files, 1):
        print(f"  {i}. {file_path.name}")
    
    print(f"  {len(files) + 1}. Back")
    
    try:
        choice = input(f"\nSelect image [1-{len(files) + 1}]: ").strip()
        choice_num = int(choice)
        
        if choice_num == len(files) + 1:
            return
        
        if 1 <= choice_num <= len(files):
            image_path = files[choice_num - 1]
            process_single_image(image_path, quality_preset, auto_enhance)
        else:
            print(f"\n{ERR} Invalid selection.")
    
    except ValueError:
        print(f"\n{ERR} Invalid input.")


def process_single_image(image_path, quality_preset, auto_enhance=False):
    """
    Main processing pipeline for single image.
    
    Args:
        image_path: Path to source image
        quality_preset: "low_quality", "medium_quality", or "high_quality"
        auto_enhance: Apply photo enhancement before processing
    """
    
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"  Processing: {image_path.name}")
    print(f"  Quality: {quality_preset.replace('_', ' ').title()}")
    print(f"{'='*60}")
    
    # Create output directory with UNIQUE name
    project_name = safe_name_from_file(image_path)
    output_dir = DIR_3D / f"{project_name}_{quality_preset}"
    
    # Ensure unique folder (add _2, _3, etc. if exists)
    counter = 2
    while output_dir.exists():
        output_dir = DIR_3D / f"{project_name}_{quality_preset}_{counter}"
        counter += 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  📁 Output folder: {output_dir.name}")
    
    # STEP 1: Photo enhancement (if from photos/ folder)
    working_image = image_path
    
    if auto_enhance:
        print(f"\n📸 Auto-enhancing photo...")
        enhanced_path = DIR_ENHANCED / f"{image_path.stem}_enhanced.png"
        
        # Check if already enhanced
        if enhanced_path.exists():
            print(f"   Using cached enhanced version")
            working_image = enhanced_path
        else:
            # Enhance and cache
            from photo_preprocess import preprocess_photo
            preset = cfg.get("auto_enhance_preset", "minimal")
            
            preprocess_photo(
                str(image_path),
                str(enhanced_path),
                preset=preset,
                save_intermediate=False
            )
            working_image = enhanced_path
            print(f"   {OK} Enhanced and cached")
    
    # STEP 2: Background removal
    if REMOVE_BACKGROUND:
        print(f"\n🎭 Removing background...")
        nobg_path = output_dir / f"{project_name}_nobg.png"
        working_image = remove_background_if_enabled(working_image, nobg_path)
        print(f"  DEBUG: After bg removal, working_image = {working_image.name}")
    
    # Copy source to output
    shutil.copy2(working_image, output_dir / "source.png")
    
    # STEP 3: Prepare for Marigold (composite transparent images)
    marigold_input = working_image
    
    if REMOVE_BACKGROUND:
        # Check if image has transparency
        img = Image.open(working_image)
        
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            print(f"  Compositing transparent image onto neutral background...")
            
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create neutral gray background (128, 128, 128)
            background = Image.new('RGB', img.size, (128, 128, 128))
            
            # Paste image onto background using alpha as mask
            background.paste(img, (0, 0), img)
            
            # VERIFY it's RGB and has valid data
            print(f"  DEBUG: Composited image mode: {background.mode}")
            print(f"  DEBUG: Composited image size: {background.size}")
            arr = np.array(background)
            print(f"  DEBUG: Pixel value range: {arr.min()} - {arr.max()}")
            print(f"  DEBUG: Mean pixel value: {arr.mean():.1f}")
            
            # Save composited version - ENSURE IT'S RGB
            prepared_path = output_dir / f"{project_name}_prepared_for_marigold.png"
            background.save(prepared_path, 'PNG')
            
            # VERIFY the saved file
            verify = Image.open(prepared_path)
            print(f"  DEBUG: Saved file mode: {verify.mode}")
            print(f"  DEBUG: Saved file size: {verify.size}")
            
            marigold_input = prepared_path
            print(f"  ✓ Prepared: {prepared_path.name}")
    
    print(f"\n  ➡️ FINAL marigold_input = {marigold_input.name}")

    # STEP 4: Generate depth map
    depth_path = output_dir / f"{project_name}_depth_16bit.png"
    marigold_opts = MARIGOLD_PRESETS[quality_preset]
    
    # DEBUG: Show what image is being sent to Marigold
    print(f"\n🔍 DEBUG INFO:")
    print(f"   Marigold input: {marigold_input.name}")
    print(f"   Input mode: {Image.open(marigold_input).mode}")
    print(f"   Input size: {Image.open(marigold_input).size}")
    
    # Use regional processing if enabled, otherwise standard
    if cfg.get('region_processing', {}).get('enabled', False):
        run_marigold_with_regions(marigold_input, depth_path, cfg)
    else:
        run_marigold_cli(marigold_input, depth_path, marigold_opts)
    
    
   # STEP 6: Extrude to 3D model
    stl_raw_path = output_dir / f"{project_name}_raw.stl"
    run_extrude_cli(depth_path, stl_raw_path, EXTRUDE_DEFAULTS)
    
    # STEP 7: Advanced post-processing (if enabled)
    if cfg.get("enable_advanced_postprocessing", False):
        print(f"\n✨ Applying advanced post-processing...")
        
        from mesh_postprocess_advanced import advanced_postprocess_pipeline
        
        stl_final_path = output_dir / f"{project_name}_final.stl"
        
        pp_settings = cfg.get("postprocessing_settings", {})
        
        advanced_postprocess_pipeline(
            str(stl_raw_path),
            str(stl_final_path),
            relief_height_mm=pp_settings.get("relief_height_mm", 10.0),
            base_thickness_mm=pp_settings.get("base_thickness_mm", 2.0),
            target_faces=pp_settings.get("target_faces", 50000),
            smoothing_iterations=pp_settings.get("smoothing_iterations", 2),
            repair=pp_settings.get("repair_mesh", True)
        )
        
        print(f"\n{OK} Post-processing complete!")
        print(f"   Raw model:   {stl_raw_path.name}")
        print(f"   Final model: {stl_final_path.name}")
    else:
        stl_final_path = stl_raw_path
        print(f"\n Post-processing disabled (enable in config.yaml)")
    
    # STEP 8: Export additional formats (if enabled)
    export_formats = cfg.get("export_formats", {"stl": True, "glb": False, "obj": False})
    
    if export_formats.get("glb", False) or export_formats.get("obj", False):
        import trimesh
        mesh = trimesh.load(stl_final_path)
        
        if export_formats.get("glb", False):
            glb_path = output_dir / f"{project_name}.glb"
            mesh.export(glb_path, file_type="glb")
            print(f"   Exported: {glb_path.name}")
        
        if export_formats.get("obj", False):
            obj_path = output_dir / f"{project_name}.obj"
            mesh.export(obj_path, file_type="obj")
            print(f"   Exported: {obj_path.name}")
    
    # Calculate total time
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    
    print(f"\n{'='*60}")
    print(f"{OK} COMPLETE!")
    print(f"   Output: {output_dir.name}/")
    print(f"   Total time: {time_str}")
    print(f"{'='*60}")
    
    # Clean up source file if configured
    if cfg.get("delete_source_after_processing", False):
        try:
            image_path.unlink()
            print(f"   {TRASH} Deleted source file")
        except:
            pass
    
    input("\nPress Enter to continue...")


def reprocess_depth_map(depth_path):
    """
    Rerun extrusion + post-processing on existing depth map.
    """
    
    print(f"\n{'='*60}")
    print(f"  Reprocessing: {depth_path.parent.name}")
    print(f"{'='*60}")
    
    output_dir = depth_path.parent
    project_name = depth_path.parent.name
    
    global cfg, EXTRUDE_DEFAULTS
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    EXTRUDE_DEFAULTS = cfg.get("extrude_defaults", {})
    print(f"✓ Loaded latest extrusion settings from config.yaml")

    # Find source image
    source_candidates = list(output_dir.glob("source.*"))
    source_image = source_candidates[0] if source_candidates else None
    
    # Extrude
    stl_raw_path = output_dir / f"{project_name}_raw_v2.stl"
    run_extrude_cli(depth_path, stl_raw_path, EXTRUDE_DEFAULTS)
    
    # Post-process if enabled
    if cfg.get("enable_advanced_postprocessing", False):
        stl_final_path = output_dir / f"{project_name}_final_v2.stl"
        
        from mesh_postprocess_advanced import advanced_postprocess_pipeline
        pp_settings = cfg.get("postprocessing_settings", {})
        
        advanced_postprocess_pipeline(
            str(stl_raw_path),
            str(stl_final_path),
            **pp_settings
        )
        
        print(f"\n{OK} Reprocessing complete!")
        print(f"   New models saved with '_v2' suffix")
    else:
        print(f"\n{OK} Reprocessing complete!")
        print(f"   New model: {stl_raw_path.name}")
    
    input("\nPress Enter to continue...")


def batch_process_folder(quality="high_quality"):
    """
    Process all images in selected folder.
    """
    
    print(f"\n{'─'*60}")
    print("BATCH PROCESS")
    print('─'*60)
    print("  1. Process all AI_files/")
    print("  2. Process all Photos/ (with enhancement)")
    print("  3. Back")
    print('─'*60)
    
    choice = input("\nSelect [1-3]: ").strip()
    
    if choice == "1":
        source_dir = DIR_AI_GENERATED
        auto_enhance = False
        print(f"\n📁 Batch processing: {source_dir.relative_to(HERE.parent)}")
    elif choice == "2":
        source_dir = DIR_PHOTOS
        auto_enhance = True
        print(f"\n📁 Batch processing: {source_dir.relative_to(HERE.parent)}")
    elif choice == "3":
        return
    else:
        print(f"\n{ERR} Invalid option.")
        return
    
    files = list_image_files(source_dir)
    
    if not files:
        print(f"\n{WARN} No images found in {source_dir.name}/")
        input("\nPress Enter to continue...")
        return
    
    print(f"\nFound {len(files)} image(s). Processing with '{quality}' preset...")
    confirm = input("Continue? [Y/n]: ").strip().lower()
    
    if confirm and confirm not in ['y', 'yes']:
        return
    
    batch_start = time.time()
    
    for i, image_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")
        try:
            process_single_image(image_path, quality, auto_enhance)
        except Exception as e:
            print(f"{ERR} Failed to process {image_path.name}: {e}")
            continue
    
    # Summary
    elapsed = time.time() - batch_start
    mins, secs = divmod(int(elapsed), 60)
    
    print(f"\n{'='*60}")
    print(f"{OK} BATCH COMPLETE")
    print(f"   Processed: {len(files)} images")
    print(f"   Total time: {mins}m {secs}s")
    print(f"{'='*60}")
    
    input("\nPress Enter to continue...")


def edit_configuration():
    """Open config.yaml in default editor."""
    
    print(f"\n{'─'*60}")
    print("EDIT CONFIGURATION")
    print('─'*60)
    print(f"Opening: {CONFIG_PATH}")
    print('─'*60)
    
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                ['cmd', '/c', 'start', '', str(CONFIG_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(CONFIG_PATH)])
        else:
            subprocess.Popen(["xdg-open", str(CONFIG_PATH)])
        
        print(f"{OK} Config opened in default editor.")
    except Exception as e:
        print(f"{WARN} Could not open automatically: {e}")
        print(f"      Please edit manually: {CONFIG_PATH}")
    
    input("\nPress Enter when done editing...")
    
    # Reload config
    global cfg
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"{OK} Configuration reloaded.")


def list_image_files(directory):
    """List all valid image files in directory."""
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = []
    
    # Make sure directory exists
    if not directory.exists():
        print(f"  ⚠️ Directory does not exist: {directory}")
        return files
    
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in valid_extensions:
            files.append(p)
    
    return sorted(files)


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")

