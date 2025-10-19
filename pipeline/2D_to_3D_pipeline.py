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

# Add scripts directory to Python path
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
SCRIPTS_DIR = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR / "depth_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "image_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "model_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "photo_preprocessing"))

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
# If conda_exe is just 'conda', try to find the full path in the system's PATH
if CONDA_EXE == "conda":
    conda_path = shutil.which("conda")
    if conda_path:
        print(f"Auto-detected conda executable at: {conda_path}")
        CONDA_EXE = conda_path
    else:
        print("\nWARNING: Could not find 'conda' in the system PATH.")
        print("         Please specify the full path to 'conda.exe' or 'conda.bat' in config.yaml if you encounter errors.")
# --- END FIX ---

MARIGOLD_ENV = cfg.get("marigold_env", "marigold")
DEPTH_ENV = cfg.get("depth_env", "depth")
MARIGOLD_CLI = (HERE / cfg.get("marigold_cli", "marigold_cli.py")).resolve()
EXTRUDE_CLI = (HERE / Path(cfg.get("extrude_cli", "../depth-to-3d-print/extrude.py"))).resolve()
DIR_2D = (HERE / cfg.get("dir_2d", "../2D_files")).resolve()
DIR_3D = (HERE / cfg.get("dir_3d", "../3D_files")).resolve()
USE_CONDA = bool(cfg.get("use_conda", True))

# Background removal settings
REMOVE_BACKGROUND = bool(cfg.get("remove_background", True))
BG_REMOVAL_METHOD = cfg.get("bg_removal_method", "removebg")  # "removebg" or "rembg"
BG_REMOVAL_MODEL = cfg.get("bg_removal_model", "isnet-general-use")  # for rembg
BG_CROP_ENABLED = bool(cfg.get("bg_crop_enabled", True))  # Auto-crop transparent borders
BG_CROP_MARGIN = int(cfg.get("bg_crop_margin", 10))  # Pixels to leave around subject
REMOVEBG_API_KEY = os.environ.get('REMOVEBG_API_KEY')  # from environment

# Load presets and defaults from config
MARIGOLD_PRESETS = cfg.get("marigold_presets", {})
EXTRUDE_DEFAULTS = cfg.get("extrude_defaults", {})

# ensure folders exist
for d in (DIR_2D, DIR_3D):
    d.mkdir(parents=True, exist_ok=True)


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
else:
    OK = "✅"
    ERR = "❌"
    WARN = "⚠️"
    TRASH = "🗑️"


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
    Uses the base filename without the timestamp suffix.
    """
    base = file_path.stem
    # Remove any existing timestamp patterns like _20251008_232149_575c44
    import re
    cleaned = re.sub(r'_\d{8}_\d{6}_[a-f0-9]{6}', '', base)
    return get_next_folder_name(cleaned, DIR_3D)

def remove_background_if_enabled(image_path: Path, output_path: Path = None) -> Path:
    """
    Remove background from image if enabled in config.
    Returns path to processed image (either cleaned or original).
    """
    if not REMOVE_BACKGROUND:
        # Background removal disabled - return original
        return image_path
    
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_nobg.png"
    
    try:
        print(f"  Removing background from {image_path.name}...")
        with Image.open(image_path) as input_img:
            output_img = remove(input_img)
            output_img.save(output_path, 'PNG')
        print(f"  {OK} Background removed (transparent)")
        return output_path
    except Exception as e:
        print(f"  {WARN} Background removal failed: {e}")
        print(f"  {WARN} Continuing with original image...")
        return image_path


def remove_background_if_enabled(image_path: Path, output_path: Path = None) -> Path:
    """
    Remove background from image if enabled in config.
    Uses remove.bg API (paid, high quality) or rembg (free).
    Automatically crops transparent borders to reduce processing time.
    Returns path to processed image (either cleaned or original).
    """
    if not REMOVE_BACKGROUND:
        return image_path
    
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_nobg.png"
    
    try:
        print(f"  Removing background from {image_path.name}...")
        
        # Choose method and remove background with cropping
        if BG_REMOVAL_METHOD == "removebg":
            remove_background(
                str(image_path), 
                str(output_path), 
                method="removebg", 
                crop=BG_CROP_ENABLED,
                margin=BG_CROP_MARGIN,
                api_key=REMOVEBG_API_KEY
            )
        else:
            remove_background(
                str(image_path), 
                str(output_path), 
                method="rembg", 
                crop=BG_CROP_ENABLED,
                margin=BG_CROP_MARGIN,
                model=BG_REMOVAL_MODEL
            )
        
        return output_path
        
    except Exception as e:
        print(f"  {WARN} Background removal failed: {e}")
        print(f"  {WARN} Continuing with original image...")
        return image_path


def generate_via_gemini(user_desc: str, filename_out: Path):
    """Call generate_with_gemini.py helper via subprocess (low quality, free)."""
    gen_py = SCRIPTS_DIR / "image_generation" / "generate_with_gemini.py"
    if not gen_py.exists():
        raise RuntimeError(f"generate_with_gemini.py not found at {gen_py}")
    
    cmd = [sys.executable, str(gen_py), "--prompt", user_desc, "--out", str(filename_out)]
    rc, output = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"Image generation failed.\n\nOutput:\n{output}")
    return filename_out


def generate_via_imagen3(user_desc: str, filename_out: Path):
    """Call generate_with_imagen3.py helper via subprocess (high quality, Imagen 3 via Vertex AI)."""
    gen_py = SCRIPTS_DIR / "image_generation" / "generate_with_imagen3.py"
    if not gen_py.exists():
        raise RuntimeError(f"generate_with_imagen3.py not found at {gen_py}")
    
    # Suppress stderr warnings from gRPC/ALTS
    cmd = [sys.executable, str(gen_py), "--prompt", user_desc, "--out", str(filename_out)]
    
    # Run with stderr suppressed for cleaner output
    import subprocess
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        # Only print stdout (our actual messages), skip stderr (gRPC warnings)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
        
        if result.returncode != 0:
            # If it failed, show the error
            if result.stderr:
                print(f"    Error output: {result.stderr}")
            raise RuntimeError(f"Image generation failed with exit code {result.returncode}")
            
        return filename_out
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Image generation failed.\n\nOutput:\n{e.output}")


def generate_image_interactive(use_high_quality=False):
    """Interactive image generation with user prompt."""
    prompt = input("\nEnter image description (or 'cancel'): ").strip()
    if prompt.lower() == "cancel":
        return
    
    # Use prompt as the base filename
    safe_prompt = "".join([c if c.isalnum() or c in ("-", "_", " ") else "_" for c in prompt])
    safe_prompt = safe_prompt.strip().replace(" ", "_")[:50]
    out_path = DIR_2D / f"{safe_prompt}.png"
    
    # Add number if file exists
    counter = 2
    while out_path.exists():
        out_path = DIR_2D / f"{safe_prompt}_{counter}.png"
        counter += 1
    
    try:
        start_time = time.time()  # START TIMING
        
        if use_high_quality:
            print(f"\nGenerating with Imagen 3 (high quality)...")
            generate_via_imagen3(prompt, out_path)
        else:
            print(f"\nGenerating with Gemini (basic quality)...")
            generate_via_gemini(prompt, out_path)
        
        # SHOW TIMING
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        quality = "high quality" if use_high_quality else "basic quality"
        print(f"\n{OK} Image saved: {out_path.name}")
        print(f"   Generation time ({quality}): {time_str}")
        
    except Exception as e:
        print(f"\n{ERR} Image generation failed: {e}")
        

def run_marigold_cli(image_path: Path, depth_out: Path, marigold_opts: dict):
    """Run marigold_cli.py to create a 16-bit depth PNG."""
    marigold_model_path = HERE / ".." / "models" / "marigold_model"
    if not marigold_model_path.exists():
        raise RuntimeError(f"Marigold model not found at {marigold_model_path}. Please run download_model.py first.")

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
    
    # Add checkpoint flags
    if marigold_opts.get("marigold_save_checkpoints", False):
        cmd.append("--save_checkpoints")
    
    if marigold_opts.get("marigold_resume", False):
        cmd.append("--resume")
        
    full = conda_prefix_cmd(MARIGOLD_ENV, cmd)
    
    print(f"\nGenerating depth map from {image_path.name}...")
    rc, output = run_cmd(full)
    
    if rc != 0:
        last_lines = "\n".join(output.splitlines()[-5:])
        raise RuntimeError(f"Marigold depth generation failed.\n\nLast output from script:\n{last_lines}")
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

    full = conda_prefix_cmd(DEPTH_ENV, cmd)
    
    print(f"\nConverting depth map to 3D model...")
    rc, output = run_cmd(full)
    
    if rc != 0:
        last_lines = "\n".join(output.splitlines()[-5:])
        raise RuntimeError(f"3D extrusion failed.\n\nLast output from script:\n{last_lines}")
    return stl_out


def list_2d_files():
    """List all valid image files in 2D_files directory."""
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = []
    for p in DIR_2D.iterdir():
        if p.is_file() and p.suffix.lower() in valid_extensions:
            files.append(p)
    return sorted(files)
    

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

        # STEP 2: Run Marigold (depth only)
        run_marigold_cli(working_image, depth_out, marigold_opts)

        # STEP 3: Mask depth map with alpha channel (remove background from depth)
        if REMOVE_BACKGROUND and working_image.suffix.lower() == '.png':
            try:
                mask_depth_with_alpha(depth_out, working_image)
            except Exception as e:
                print(f"  {WARN} Could not mask depth map: {e}")
                print(f"  {WARN} Continuing with unmasked depth...")

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


def select_and_process_single(preset_key: str):
    """Helper function to list files, prompt for selection, and process one image."""
    files = list_2d_files()
    if not files:
        print("\nNo images found in 2D_files/.")
        return
    
    print("\nAvailable images:")
    for i, p in enumerate(files, start=1):
        print(f"  {i}) {p.name}")
    
    sel = input(f"\nPick file [1-{len(files)}] or 'c' to cancel: ").strip()
    if sel.lower() == "c":
        return
    
    try:
        target = files[int(sel) - 1]
        preset = MARIGOLD_PRESETS[preset_key]
        print(f"\nProcessing '{target.name}' with '{preset_key}' preset...")
        process_single(target, preset, EXTRUDE_DEFAULTS, quality_preset=preset_key)  # PASS PRESET NAME
    except (ValueError, IndexError):
        print("Invalid selection.")


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


def interactive_loop():
    print(f"\n{'='*60}\n  Marigold -> Depth-to-3D Pipeline\n{'='*60}")
    
    while True:
        print("\n" + "-"*60 + "\nMENU:")
        print("  1) Generate new image (basic quality)")
        print("  2) Generate new image (high quality, Imagen 3)")
        print("  3) Enhance photos (auto-detect settings)")           # NEW
        print("  4) Enhance single photo (preview)")                  # NEW
        print("  5) Batch process all images (high quality)")
        print("  6) Process single image (low quality)")
        print("  7) Process single image (medium quality)")
        print("  8) Process single image (high quality)")
        print("  9) Edit default parameters (opens config.yaml)")
        print(" 10) Quit\n" + "-"*60)
        
        choice = input("Select option [1-10]: ").strip()
        
        if choice == "1":
            generate_image_interactive(use_high_quality=False)
                
        elif choice == "2":
            generate_image_interactive(use_high_quality=True)
            
        elif choice == "3":
            enhance_photos_batch()
            
        elif choice == "4":
            enhance_single_photo_interactive()
            
        elif choice == "5":
            files = list_2d_files()
            if not files:
                print("No images found. Generate or enhance some first.")
                continue
            
            print(f"\nFound {len(files)} image(s). Processing with 'high_quality' preset.")
            batch_start = time.time()
            for i, p in enumerate(files, start=1):
                process_single(p, MARIGOLD_PRESETS['high_quality'], EXTRUDE_DEFAULTS, f"{i}/{len(files)}", quality_preset="high")  # ADD QUALITY
            
            mins, secs = divmod(int(time.time() - batch_start), 60)
            print(f"\n{'='*60}\n{OK} BATCH COMPLETE: Processed {len(files)} file(s) in {mins}m {secs}s\n{'='*60}")
            
        elif choice == "6":
            select_and_process_single('low_quality')
            
        elif choice == "7":
            select_and_process_single('medium_quality')
            
        elif choice == "8":
            select_and_process_single('high_quality')

        elif choice == "9":
            view_edit_defaults()
            
        elif choice == "10":
            print("\nExiting. Goodbye!")
            break
        else:
            print("\nInvalid option.")


def enhance_photos_batch():
    """
    Enhance all photos in 2D_files using automatic detection.
    Includes background removal if enabled in config.
    """
    from photo_analyzer import analyze_photo
    
    files = list_2d_files()
    if not files:
        print("\nNo images found in 2D_files/.")
        return
    
    print(f"\n{'='*60}")
    print(f"  Photo Enhancement - Automatic Detection")
    print(f"{'='*60}")
    
    if REMOVE_BACKGROUND:
        print(f"  Background removal: ENABLED")
    else:
        print(f"  Background removal: DISABLED")
    
    print(f"\nFound {len(files)} image(s) to enhance.\n")
    
    batch_start = time.time()
    
    # Create enhanced directory
    enhanced_dir = DIR_2D.parent / "data" / "2D_files_enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    log_path = enhanced_dir / "enhancement_log.txt"
    
    with open(log_path, 'w') as log:
        log.write("Photo Enhancement Log\n")
        log.write("="*60 + "\n\n")
        
        for i, img_path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Processing: {img_path.name}")
            
            try:
                # STEP 1: Remove background if enabled
                working_image = img_path
                if REMOVE_BACKGROUND:
                    nobg_path = enhanced_dir / f"{img_path.stem}_nobg_temp.png"
                    working_image = remove_background_if_enabled(img_path, nobg_path)
                
                # STEP 2: Analyze photo
                settings, reasons, cmd = analyze_photo(str(working_image), verbose=False)
                
                # Show brief analysis
                print(f"  Detected: {settings['preset']} enhancement needed")
                for reason in reasons[:2]:  # Show top 2 reasons
                    print(f"    • {reason}")
                
                # STEP 3: Enhance photo
                output_path = enhanced_dir / f"{img_path.stem}_enhanced.png"
                
                from photo_preprocess import preprocess_photo
                
                preprocess_photo(
                    str(working_image),
                    str(output_path),
                    preset=settings['preset'],
                    denoise_strength=settings['denoise_strength'],
                    use_hdr=settings['use_hdr'],
                    clahe_clip=settings['clahe_clip'],
                    sharpen_percent=settings['sharpen_percent'],
                    saturation=settings['saturation'],
                    enhance_details=settings['detail_amount'] if settings['use_detail_enhancement'] else 0,
                    save_intermediate=False
                )
                
                print(f"  {OK} Enhanced: {output_path.name}")
                print(f"  To manually adjust: {cmd}\n")
                
                # Log the command
                log.write(f"File: {img_path.name}\n")
                log.write(f"Background removed: {REMOVE_BACKGROUND}\n")
                log.write(f"Settings: {settings['preset']}\n")
                log.write(f"Command: {cmd}\n")
                log.write("-" * 60 + "\n\n")
                
                # Clean up temp nobg file
                if REMOVE_BACKGROUND and working_image != img_path and working_image.exists():
                    working_image.unlink()
                
            except Exception as e:
                print(f"  {ERR} Failed: {e}\n")
                import traceback
                traceback.print_exc()
                log.write(f"File: {img_path.name}\n")
                log.write(f"ERROR: {e}\n")
                log.write("-" * 60 + "\n\n")
    
    # Show total time
    elapsed = time.time() - batch_start
    mins, secs = divmod(int(elapsed), 60)
    time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    
    print(f"\n{OK} Enhancement complete!")
    print(f"   Total time: {time_str}")
    print(f"   Enhanced images: {enhanced_dir}")
    print(f"   Settings log: {log_path}")
    print(f"\nNext steps:")
    print(f"  1. Review enhanced images in: {enhanced_dir.name}/")
    print(f"  2. If satisfied, move them to 2D_files/ and delete originals")
    print(f"  3. If adjustments needed, use commands from: {log_path.name}")
    print(f"  4. Run main pipeline to process enhanced images\n")


def enhance_single_photo_interactive():
    """
    Enhance a single photo with interactive selection.
    Includes background removal if enabled in config.
    """
    from photo_analyzer import analyze_photo
    from photo_preprocess import preprocess_photo
    
    files = list_2d_files()
    if not files:
        print("\nNo images found in 2D_files/.")
        return
    
    print("\nAvailable images:")
    for i, p in enumerate(files, start=1):
        print(f"  {i}) {p.name}")
    
    sel = input(f"\nPick file [1-{len(files)}] or 'c' to cancel: ").strip()
    if sel.lower() == 'c':
        return
    
    try:
        img_path = files[int(sel) - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    print(f"\n{'='*60}")
    print(f"  Analyzing: {img_path.name}")
    print(f"{'='*60}")
    
    if REMOVE_BACKGROUND:
        print(f"  Background removal: ENABLED")
    else:
        print(f"  Background removal: DISABLED")
    
    start_time = time.time()
    
    try:
        # STEP 1: Remove background if enabled
        working_image = img_path
        if REMOVE_BACKGROUND:
            nobg_path = img_path.parent / f"{img_path.stem}_nobg_temp.png"
            working_image = remove_background_if_enabled(img_path, nobg_path)
        
        # STEP 2: Analyze with full output
        settings, reasons, cmd = analyze_photo(str(working_image), verbose=True)
        
        # Ask for confirmation
        confirm = input("\nProceed with these settings? [Y/n]: ").strip().lower()
        if confirm and confirm not in ['y', 'yes']:
            print("Enhancement cancelled.")
            # Clean up temp file
            if REMOVE_BACKGROUND and working_image != img_path and working_image.exists():
                working_image.unlink()
            return
        
        # STEP 3: Enhance
        output_path = img_path.parent / f"{img_path.stem}_enhanced.png"
        
        print(f"\nEnhancing...")
        preprocess_photo(
            str(working_image),
            str(output_path),
            preset=settings['preset'],
            denoise_strength=settings['denoise_strength'],
            use_hdr=settings['use_hdr'],
            clahe_clip=settings['clahe_clip'],
            sharpen_percent=settings['sharpen_percent'],
            saturation=settings['saturation'],
            enhance_details=settings['detail_amount'] if settings['use_detail_enhancement'] else 0,
            save_intermediate=False
        )
        
        # Clean up temp nobg file
        if REMOVE_BACKGROUND and working_image != img_path and working_image.exists():
            working_image.unlink()
        
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        print(f"\n{OK} Enhanced image saved: {output_path.name}")
        print(f"   Enhancement time: {time_str}")
        if REMOVE_BACKGROUND:
            print(f"   Background: Removed (transparent)")
        print(f"\nTo adjust manually:")
        print(f"  {cmd}\n")
        
    except Exception as e:
        print(f"\n{ERR} Enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp file on error
        if REMOVE_BACKGROUND and 'working_image' in locals() and working_image != img_path and working_image.exists():
            working_image.unlink()


if __name__ == "__main__":
    try:
        interactive_loop()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")

