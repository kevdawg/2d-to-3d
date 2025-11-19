#!/usr/bin/env python3
"""
Interactive launcher for 2D to 3D Pipeline with timing tracking.
- Uses conda run to execute scripts in their respective conda environments
- Includes CLI argument support for automated testing
- Comprehensive timing and performance profiling
"""

import argparse
from timing_tracker import TimingTracker
import os
import sys
from pathlib import Path
import time
import shutil
import subprocess
import argparse
import yaml
import platform
from PIL import Image
import numpy as np
import scipy.stats as st
import json
import re
import unicodedata
from pprint import pprint

# Add timing tracker
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from timing_tracker import TimingTracker

CONFIG_PATH = HERE / "config.yaml"
SCRIPTS_DIR = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR / "depth_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "image_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "model_generation"))
sys.path.insert(0, str(SCRIPTS_DIR / "photo_preprocessing"))

# Import background removal functions
#from background_remover_removebg import remove_background
from depth_masking import mask_depth_with_alpha

if not CONFIG_PATH.exists():
    print(f"❌ Missing config.yaml in pipeline folder. Create it from the example.")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Config values with defaults
CONDA_EXE = cfg.get("conda_exe", "conda")
GOOGLE_CLOUD_PROJECT = cfg.get("google_cloud_project", "gen-lang-client-0248735054") # <-- NEW

# Auto-detect full conda path
if CONDA_EXE == "conda":
    conda_path = shutil.which("conda")
    if conda_path:
        print(f"Auto-detected conda executable at: {conda_path}")
        CONDA_EXE = conda_path

# Get model paths from config
models_config = cfg.get("models", {})
MARIGOLD_MODEL_PATH = (HERE / ".." / models_config.get("marigold", "models/marigold_model")).resolve()
REALESRGAN_MODELS_DIR = (HERE / ".." / models_config.get("realesrgan", "models/RealESRGAN")).resolve()
REMBG_MODELS_DIR = (HERE / ".." / models_config.get("rembg", "models/rembg_models")).resolve()
GFPGAN_MODELS_DIR = (HERE / ".." / models_config.get("gfpgan", "models/GFPGAN")).resolve() # <-- NEW

# Ensure directories exist
for model_dir in [MARIGOLD_MODEL_PATH.parent, REALESRGAN_MODELS_DIR, REMBG_MODELS_DIR, GFPGAN_MODELS_DIR]: # <-- UPDATED
    model_dir.mkdir(parents=True, exist_ok=True)

# ONLY Rembg requires environment variable (library limitation)
os.environ['U2NET_HOME'] = str(REMBG_MODELS_DIR)


MARIGOLD_ENV = cfg.get("marigold_env", "marigold")
DEPTH_ENV = cfg.get("depth_env", "depth-to-3d")
IMAGEN_ENV = cfg.get("imagen_env", "aigen") # <-- Use 'aigen'
PHOTO_PREP_ENV = cfg.get("photo_prep_env", "photo-prep")
DIR_AI = (HERE / cfg.get("dir_ai_generated", "../data/AI_files")).resolve()
DIR_PHOTOS = (HERE / cfg.get("dir_photos", "../data/Photos")).resolve()
DIR_3D = (HERE / cfg.get("dir_3d", "../data/3D_files")).resolve()
DIR_PHOTOS_ENHANCED = (HERE / cfg.get("dir_enhanced", "../data/Photos_enhanced")).resolve()
USE_CONDA = bool(cfg.get("use_conda", True))

# Background removal settings
REMOVE_BACKGROUND = bool(cfg.get("remove_background", True))
BG_REMOVAL_METHOD = cfg.get("bg_removal_method", "removebg")
BG_REMOVAL_MODEL = cfg.get("bg_removal_model", "isnet-general-use")
BG_CROP_ENABLED = bool(cfg.get("bg_crop_enabled", True))
BG_CROP_MARGIN = int(cfg.get("bg_crop_margin", 10))
REMOVEBG_API_KEY = os.environ.get('REMOVEBG_API_KEY')

# CLI scripts
MARIGOLD_CLI = (HERE / cfg.get("marigold_cli", "../scripts/depth_generation/marigold_cli.py")).resolve()
EXTRUDE_CLI = (HERE / cfg.get("extrude_cli", "../scripts/model_generation/extrude_cli.py")).resolve()
AI_REPAIR_CLI = (HERE / cfg.get("ai_repair_cli", "../scripts/image_generation/enhance_with_ai_prompt.py")).resolve() # <-- Use new name

# Get Tiled CLI path
TILED_DEPTH_SETTINGS = cfg.get("tiled_depth_settings", {})
MARIGOLD_TILED_CLI = (HERE / TILED_DEPTH_SETTINGS.get("marigold_tiled_cli", "../scripts/depth_generation/marigold_tiled_cli.py")).resolve()

# Load presets and defaults from config
MARIGOLD_PRESETS = cfg.get("marigold_presets", {})
EXTRUDE_DEFAULTS = cfg.get("extrude_defaults", {})

# Ensure folders exist
print("Initializing directories...")
for d, name in [(DIR_AI, "AI_files/"), (DIR_PHOTOS, "Photos/"), (DIR_3D, "3D_files/"), (DIR_PHOTOS_ENHANCED, "Photos_enhanced/")]:
    d.mkdir(parents=True, exist_ok=True)

OK = "[OK]"
ERR = "[X]"
WARN = "[!]"
TRASH = "[DEL]"
INFO = "[i]"

def run_cmd(cmd_list, show_timer=False, timer_message="Processing", cwd=None, clean_env=False):
    """
    Run a subprocess command with clean single-line progress bar display.
    Accepts cwd and a clean_env flag.
    """
    import time
    
    env_vars = None  # Default
    if clean_env:
        print("   (Executing with minimal 'PATH' environment to prevent conflicts)")
        env_vars = {}
        
        # Copy only the essential variables from the host
        essential_vars = [
            # System variables
            'PATH', 'SystemRoot', 'SYSTEMDRIVE', 'ComSpec', 'TEMP', 'TMP', 
            'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE', 
            
            # Home variables
            'USERPROFILE', 'HOME', 'HOMEDRIVE', 'HOMEPATH',
            'APPDATA', # <-- ADDED FOR WINDOWS DEFAULT AUTH

            # We MUST preserve the Google Auth variable
            'GOOGLE_APPLICATION_CREDENTIALS', # For Imagen
            'GEMINI_API_KEY',                 # For old Gemini (if still used)
            'GOOGLE_CLOUD_PROJECT',           # For Vertex AI (Imagen AND new Gemini)
            
            # Conda variables (CRITICAL for conda.bat to function)
            'CONDA_EXE', 'CONDA_ROOT', 'CONDA_SHLVL', 'CONDA_BAT',
            'CONDA_DEFAULT_ENV', 'CONDA_PREFIX' 
        ]

        # Also copy ANY other CONDA_ variables from the parent
        for var in os.environ:
            if var.startswith('CONDA_') and var not in essential_vars:
                env_vars[var] = os.environ[var]

        for var in essential_vars:
            if var in os.environ and var not in env_vars:
                env_vars[var] = os.environ[var]
        
        # Ensure PATH exists, even if minimal
        if 'PATH' not in env_vars:
            env_vars['PATH'] = os.environ.get('PATH', '')
    
    else:
        # Not a clean_env, so just copy the parent environment
        env_vars = os.environ.copy()

    # Force the subprocess (conda) to use UTF-8 for its stdout/stderr
    # This prevents the 'charmap' codec error when it prints
    # Unicode characters (like progress bars) to the console.
    env_vars['PYTHONIOENCODING'] = 'utf-8'
            
    try:
        output_lines = []
        proc = subprocess.Popen(
            cmd_list, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace', 
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            bufsize=1,  # Line buffered
            cwd=cwd,      # <-- SETS THE CURRENT WORKING DIRECTORY
            env=env_vars  # <-- SETS THE CLEAN ENVIRONMENT
        )
        
        start_time = time.time()
        last_progress_line = None
        
        # (Rest of the function is unchanged)
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
                
            line = line.rstrip('\n\r')
            output_lines.append(line)
            
            is_progress = '%|' in line or 'it/s' in line or 'it]' in line
            
            if is_progress:
                sys.stdout.write('\r' + ' ' * 100 + '\r')
                sys.stdout.write('    ' + line)
                sys.stdout.flush()
                last_progress_line = line
            else:
                if last_progress_line:
                    sys.stdout.write('\n')
                    last_progress_line = None
                sys.stdout.write('    ' + line + '\n')
                sys.stdout.flush()
        
        proc.wait()
        
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


def parse_cli_args():
    """Parse command-line arguments for automated testing."""
    parser = argparse.ArgumentParser(
        description='2D to 3D Pipeline - Automated testing support',
        add_help=True
    )
    
    # Test mode
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in automated test mode (skip interactive menu)')
    parser.add_argument('--input', type=str,
                        help='Input image path (required for test mode)')
    parser.add_argument('--quality', type=str, 
                        choices=['ultra_low', 'low', 'medium', 'high', 'ultra_high'],
                        help='Quality preset (ultra_low, low, medium, high, ultra_high)')
    parser.add_argument('--resume-work-dir', type=str,
                        help='Path to an existing output folder to resume processing')
    # Marigold overrides
    parser.add_argument('--steps', type=int,
                        help='Override marigold_steps')
    parser.add_argument('--ensemble', type=int,
                        help='Override marigold_ensemble')
    parser.add_argument('--resolution', type=int,
                        help='Override marigold_processing_res')
    
    # Extrusion overrides
    parser.add_argument('--emboss', type=float,
                        help='Override emboss depth (0.1-0.8)')
    parser.add_argument('--smoothing', type=int,
                        help='Override smoothing (odd number: 1,3,5,7,9)')
    parser.add_argument('--near', type=float,
                        help='Override near_offset (0.0-1.0)')
    parser.add_argument('--far', type=float,
                        help='Override far_offset (0.0-1.0)')
    parser.add_argument('--width', type=float,
                        help='Override width_mm')
    
    # Processing options
    parser.add_argument('--no-bg-removal', action='store_true',
                        help='Skip background removal')
    parser.add_argument('--no-enhancement', action='store_true',
                        help='Skip AI enhancement')
    
    return parser.parse_args()


def apply_cli_overrides(preset_dict, extrude_dict, args):
    """
    Apply CLI argument overrides to config dictionaries.
    
    Args:
        preset_dict: Marigold preset dictionary (will be modified)
        extrude_dict: Extrusion settings dictionary (will be modified)
        args: Parsed CLI arguments
    
    Returns:
        Tuple of (modified_preset, modified_extrude)
    """
    # Deep copy to avoid modifying original
    preset = preset_dict.copy()
    extrude = extrude_dict.copy()
    
    # Marigold overrides
    if args.steps is not None:
        preset['marigold_steps'] = args.steps
    if args.ensemble is not None:
        preset['marigold_ensemble'] = args.ensemble
    if args.resolution is not None:
        preset['marigold_processing_res'] = args.resolution
    
    # Extrusion overrides
    if args.emboss is not None:
        extrude['emboss'] = args.emboss
    if args.smoothing is not None:
        extrude['smoothing'] = args.smoothing
    if args.near is not None:
        extrude['near_offset'] = args.near
    if args.far is not None:
        extrude['far_offset'] = args.far
    if args.width is not None:
        extrude['width_mm'] = args.width
    
    return preset, extrude


def log_command_to_file(output_dir: Path, command_name: str, cmd_list: list, description: str = ""):
    """
    Log a command to the project's command history file.
    
    Args:
        output_dir: Project output directory
        command_name: Name of command (e.g., "marigold", "extrude")
        cmd_list: Full command as list
        description: Optional description
    """
    log_file = output_dir / "commands.txt"
    
    # Convert command list to properly quoted string
    quoted_cmd = []
    for part in cmd_list:
        part_str = str(part)
        # Quote paths and arguments with spaces
        if ' ' in part_str or '\\' in part_str:
            quoted_cmd.append(f'"{part_str}"')
        else:
            quoted_cmd.append(part_str)
    
    cmd_string = ' '.join(quoted_cmd)
    
    # Append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        if description:
            f.write(f"\n# {description}\n")
        f.write(f"# {command_name.upper()}\n")
        f.write(f"{cmd_string}\n")
    
    print(f"  {INFO} Command logged to: {log_file.name}")

def conda_prefix_cmd(env_name, cmd_list):
    """Return a full command list that runs cmd_list inside conda env."""
    #return [CONDA_EXE, "run", "-n", env_name, "--no-capture-output"] + cmd_list if USE_CONDA else cmd_list
    return [CONDA_EXE, "run", "-n", env_name] + cmd_list if USE_CONDA else cmd_list



def conda_prefix_cmd_new(env_name, cmd_list):
    """
    Return a full command list that runs cmd_list inside a properly
    activated conda environment.
    """
    if not USE_CONDA:
        return cmd_list
    
    # Build the command string to be run (e.g., "python marigold_cli.py ...")
    quoted_cmd_parts = []
    for part in cmd_list:
        part_str = str(part)
        # Add quotes if it has a space and isn't already quoted
        if ' ' in part_str and not (part_str.startswith('"') and part_str.endswith('"')):
            quoted_cmd_parts.append(f'"{part_str}"')
        else:
            quoted_cmd_parts.append(part_str)
    run_string = ' '.join(quoted_cmd_parts)
    
    if platform.system() == "Windows":
        # On Windows, we use `cmd.exe /C` to chain commands.
        conda_bat = str(CONDA_EXE).strip('\"\'')
        
        # --- THIS IS THE FIX ---
        # Only add quotes to the path if it contains a space.
        if ' ' in conda_bat:
            conda_call = f'call "{conda_bat}"'
        else:
            conda_call = f'call {conda_bat}'
        # --- END FIX ---
            
        # The full command: call the activate script, AND THEN (&&) run our command
        full_command_string = f'{conda_call} activate {env_name} && {run_string}'
        
        # Popen expects a list: ["cmd.exe", "/C", "the entire command string"]
        return ["cmd.exe", "/C", full_command_string]
    
    else:
        # On Linux/macOS, we can use `bash -c`
        conda_base = Path(CONDA_EXE).parent.parent
        bash_init = conda_base / "etc" / "profile.d" / "conda.sh"
        
        # Add quotes for safety on Linux
        conda_call = f'source "{bash_init}" && conda'
        
        full_command_string = f'{conda_call} activate {env_name} && {run_string}'
        
        # Popen expects a list: ["/bin/bash", "-c", "the entire command string"]
        return ["/bin/bash", "-c", full_command_string]


def remove_background_if_enabled(image_path: Path, output_path: Path = None, padding: int = 0) -> Path:
    """
    Remove background from image if enabled in config.
    Uses rembg library via subprocess in the photo-prep environment.
    """
    if not REMOVE_BACKGROUND:
        return image_path
    
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_nobg.png"
    
    try:
        print(f"  Removing background from {image_path.name}...")
        
        # Get path to the script
        rembg_script = SCRIPTS_DIR / "photo_preprocessing" / "background_remover_removebg.py"
        
        # Build command for subprocess
        cmd = [
            "python", str(rembg_script),
            "--input", str(image_path),
            "--output", str(output_path),
            "--method", BG_REMOVAL_METHOD,
            "--model", BG_REMOVAL_MODEL,
            "--padding", str(padding)
        ]
        
        if BG_CROP_ENABLED:
            cmd.extend(["--margin", str(BG_CROP_MARGIN)])
        else:
            cmd.append("--no-crop")
            
        # Add advanced settings if they exist in config
        if cfg.get('bg_removal_model_secondary'):
            cmd.extend(["--model-secondary", cfg.get('bg_removal_model_secondary')])
        
        if cfg.get('bg_model_combine_mode'):
            cmd.extend(["--combine-mode", cfg.get('bg_model_combine_mode')])
            
        if cfg.get('bg_alpha_matting'):
            cmd.append("--alpha-matting")
            cmd.extend(["--matting-fg", str(cfg.get('bg_matting_foreground_threshold', 240))])
            cmd.extend(["--matting-bg", str(cfg.get('bg_matting_background_threshold', 10))])

        # Run in photo-prep environment
        full_cmd = conda_prefix_cmd(PHOTO_PREP_ENV, cmd)
        
        # Execute
        rc, output = run_cmd(full_cmd)
        
        if rc == 0 and output_path.exists():
            print(f"  {OK} Background removed (transparent)")
            return output_path
        else:
            print(f"  {WARN} Background removal failed (exit code {rc})")
            return image_path
        
    except Exception as e:
        print(f"  {WARN} Background removal failed: {e}")
        print(f"  {WARN} Continuing with original image...")
        return image_path


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


def generate_via_gemini(user_desc: str, filename_out: Path):
    """Call generate_with_gemini.py helper via subprocess in aigen conda environment."""
    gen_py = SCRIPTS_DIR / "image_generation" / "generate_with_gemini.py"
    if not gen_py.exists():
        raise RuntimeError(f"generate_with_gemini.py not found at {gen_py}")

    # Get aigen environment name from config
    aigen_env = cfg.get("imagen_env", "aigen")

    # --- START OF FIX ---
    # Revert to the original, working Popen logic that mirrors imagen3
    cmd = ["python", str(gen_py), "--prompt", user_desc, "--out", str(filename_out)]
    full_cmd = conda_prefix_cmd(aigen_env, cmd)

    try:
        # Set environment variables to suppress gRPC warnings
        env = os.environ.copy()
        env['GRPC_VERBOSITY'] = 'ERROR'
        env['GLOG_minloglevel'] = '2'
        
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            cwd=gen_py.parent, # Run from the script's directory
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        stdout, stderr = proc.communicate()
        
        # Print stdout
        if stdout:
            for line in stdout.strip().split('\n'):
                if line.strip():
                    print(f"    {line}")
        
        if proc.returncode != 0:
            # Include stderr in the error message
            error_message = f"Image generation failed with exit code {proc.returncode}"
            if stderr:
                error_message += f"\n--- Error from script ---\n{stderr.strip()}\n-------------------------"
            raise RuntimeError(error_message)
        
        return filename_out
        
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")


def generate_via_imagen3(user_desc: str, filename_out: Path):
    """Call generate_with_imagen3.py helper via subprocess in aigen conda environment."""
    gen_py = SCRIPTS_DIR / "image_generation" / "generate_with_imagen3.py"
    if not gen_py.exists():
        raise RuntimeError(f"generate_with_imagen3.py not found at {gen_py}")
    
    # Get aigen environment name from config
    aigen_env = cfg.get("imagen_env", "aigen")
    
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
            stderr=subprocess.PIPE, # <-- FIX: Capture stderr instead of DEVNULL
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        stdout, stderr = proc.communicate()
        
        # Print stdout (this is what creates the multi-line output)
        if stdout:
            for line in stdout.strip().split('\n'):
                if line.strip():
                    print(f"    {line}")
        
        if proc.returncode != 0:
            # --- FIX: Include stderr in the error message ---
            error_message = f"Image generation failed with exit code {proc.returncode}"
            if stderr:
                error_message += f"\n--- Error from script ---\n{stderr.strip()}\n-------------------------"
            raise RuntimeError(error_message)
            # --- END OF FIX ---
        
        return filename_out
        
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")


# ============================================
# INTERACTIVE WRAPPERS
# ============================================

def generate_ai_image_menu():
    """
    --- UPDATED ---
    Submenu for AI image generation.
    Selects Model *first*, then Prompt Style.
    """
    
    print(f"\n{'─'*60}")
    print("SELECT AI MODEL")
    print('─'*60)
    print("  1. Gemini (FREE, basic quality)")
    print("  2. Imagen 3 ($0.01, high quality)")
    print("  3. Back to main menu")
    print('─'*60)
    
    model_choice = input("\nSelect model [1-3]: ").strip()
    
    model_name = ""
    if model_choice == "1":
        model_name = "gemini"
    elif model_choice == "2":
        model_name = "imagen"
    elif model_choice == "3":
        return
    else:
        print(f"\n{ERR} Invalid option")
        return
    
    # Load prompts and show style submenu
    prompts_data = load_prompts()
    prompt_style_submenu(model_name, prompts_data)


def prompt_style_submenu(model_name: str, prompts_data: dict):
    """
    --- NEW FUNCTION ---
    Shows the prompt style selection menu *after* a model is chosen.
    """
    
    # Load prompts
    prompts = prompts_data["prompts"]
    default_prompt = prompts_data.get("default_prompt", "side_profile")
    allow_custom = cfg.get("allow_custom_prompts", True)
    
    while True:
        print(f"\n{'─'*60}")
        print(f"SELECT PROMPT STYLE (for {model_name.title()})")
        print('─'*60)
        
        # List available prompt styles
        prompt_keys = list(prompts.keys())
        for i, key in enumerate(prompt_keys, 1):
            prompt = prompts[key]
            marker = " (Default)" if key == default_prompt else ""
            print(f"  {i}. {prompt['name']}{marker}")
            print(f"     └─ {prompt['description']}")
        
        # Add custom option if enabled
        custom_option = len(prompt_keys) + 1
        back_option = custom_option + 1
        
        if allow_custom:
            custom_info = prompts_data.get("custom_prompt_template", {})
            print(f"  {custom_option}. {custom_info.get('name', 'Custom Prompt')}")
            print(f"     └─ {custom_info.get('description', 'Enter your own description')}")
        
        print(f"  {back_option}. Back to model selection")
        print('─'*60)
        
        choice = input(f"\nSelect prompt style [1-{back_option}]: ").strip()
        
        try:
            choice_num = int(choice)
            
            # Back
            if choice_num == back_option:
                return  # Goes back to generate_ai_image_menu()
            
            # Custom prompt
            if allow_custom and choice_num == custom_option:
                selected_style = "custom"
            # Standard prompt
            elif 1 <= choice_num <= len(prompt_keys):
                selected_style = prompt_keys[choice_num - 1]
            else:
                print(f"\n{ERR} Invalid selection")
                continue
            
            # Now get subject and generate
            # Pass model_name down
            generate_with_prompt_style(selected_style, model_name, prompts_data)
            
        except ValueError:
            print(f"\n{ERR} Invalid input")


def generate_with_prompt_style(prompt_style: str, model_name: str, prompts_data: dict):
    """
    --- UPDATED ---
    Generate image with selected prompt style and pre-selected model.
    Removes confirmation prompt.
    Handles batch syntax, e.g., "jumping frog (4)"
    
    Args:
        prompt_style: Selected prompt key or "custom"
        model_name: "gemini" or "imagen"
        prompts_data: Loaded prompts configuration
    """
    print(f"\n{'─'*60}")
    if prompt_style == "custom":
        print("CUSTOM PROMPT")
    else:
        prompt_info = prompts_data["prompts"][prompt_style]
        print(f"{prompt_info['name'].upper()}")
    print('─'*60)
    
    # Get subject description
    if prompt_style == "custom":
        subject_input = input("\nEnter full prompt (or 'cancel'): ").strip()
    else:
        subject_input = input("\nEnter subject description (e.g., 'jumping frog (4)') or 'cancel': ").strip()
    
    if subject_input.lower() == "cancel":
        return
    
    # Default values
    subject_clean = subject_input
    quantity = 1
    
    # Check for batch syntax like "a frog (4)"
    match = re.match(r'(.+)\s*\((\d+)\)\s*$', subject_input)
    
    if match:
        subject_clean = match.group(1).strip()  # "a frog"
        try:
            quantity = int(match.group(2))  # 4
            if quantity < 1:
                quantity = 1
            elif quantity > 20:  # Safety cap
                print(f"  {WARN} Batch size capped at 20.")
                quantity = 20
        except ValueError:
            quantity = 1
        
        print(f"  {INFO} Batch detected: Generating {quantity} images for prompt '{subject_clean}'")
    
    # Build full prompt using the *cleaned* subject
    full_prompt = build_full_prompt(subject_clean, prompt_style, prompts_data)
    
    # Show preview of what will be generated
    print(f"\n{INFO} Generated prompt preview:")
    preview = full_prompt[:150] + "..." if len(full_prompt) > 150 else full_prompt
    print(f"   {preview}")
    
    # Generate image(s) in a loop
    for i in range(quantity):
        if quantity > 1:
            print(f"\n--- Generating image {i + 1} of {quantity} ---")
            
        # This function already handles unique file naming,
        # so calling it in a loop will create frog.png, frog_2.png, etc.
        generate_image_interactive(full_prompt, model_name, subject_clean)

def generate_image_interactive(full_prompt: str, model: str, subject_desc: str):
    """
    Generate image with specified model and prompt.
    
    Args:
        full_prompt: Complete formatted prompt
        model: "gemini" or "imagen"
        subject_desc: The simple subject description (for the filename)
    """

    # --- THIS IS THE FIX ---

    # 1. Clean the subject string into a base name
    #    (This part from my last suggestion was correct)
    base_name = unicodedata.normalize('NFKD', subject_desc).encode('ascii', 'ignore').decode('utf-8')
    base_name = re.sub(r'[^a-zA-Z0-9\s-]', '', base_name).strip()
    base_name = re.sub(r'\s+', '_', base_name).lower()
    
    # Truncate if it was a long custom prompt. This is what causes
    # the long name you saw.
    base_name = base_name[:50].strip('_') 

    if not base_name: # Fallback
        base_name = "ai_generated"

    # 2. Use YOUR original, correct uniqueness check.
    #    This logic correctly checks for the file *with* the .png extension.
    out_path = DIR_AI / f"{base_name}.png"
    counter = 2
    while out_path.exists():
        out_path = DIR_AI / f"{base_name}_{counter}.png"
        counter += 1
    # --- END FIX ---
    
    try:
        start_time = time.time()
        
        print(f"\nGenerating with {model.title()}...")
        
        if model == "gemini":
            generate_via_gemini(full_prompt, out_path)
        else:  # imagen
            generate_via_imagen3(full_prompt, out_path)
        
        # Show timing
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        print(f"\n{OK} Image saved: {out_path.name}")
        print(f"   Generation time: {time_str}")
        if model == "imagen":
            print(f"   Cost: $0.01")
        
    except Exception as e:
        print(f"\n{ERR} Image generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


# --- NEW FUNCTION TO RUN AI REPAIR ---
def run_ai_repair_cli(input_path: Path, output_path: Path, prompt: str, tracker: TimingTracker):
    """Call enhance_with_ai_prompt.py to repair an image."""
    
    if not AI_REPAIR_CLI.exists():
        raise RuntimeError(f"AI Repair CLI not found at {AI_REPAIR_CLI}. Check config.yaml.")

    tracker.substep("Initializing Generative AI Repair")
    
    # Set the CWD where the script will run
    script_cwd = AI_REPAIR_CLI.parent

    # Calculate all paths RELATIVE to that CWD.
    rel_input_path = os.path.relpath(input_path, script_cwd)
    rel_output_path = os.path.relpath(output_path, script_cwd)

    # 1. This command is for EXECUTION (relative paths)
    cmd_for_exec = ["python", "-u", AI_REPAIR_CLI.name,
           "--input", rel_input_path,
           "--output", rel_output_path,
           "--project", str(cfg.get("google_cloud_project", GOOGLE_CLOUD_PROJECT)),
           "--prompt"  # The flag
           ]
    # --- FIX: Add each word of the prompt as a separate item ---
    cmd_for_exec.extend(str(prompt).split())
    
    # 2. This command is for LOGGING (absolute paths)
    # --- FIX: Rebuild the log command correctly ---
    cmd_for_log = ["python", "-u", str(AI_REPAIR_CLI),
           "--input", str(input_path),
           "--output", str(output_path),
           "--project", str(cfg.get("google_cloud_project", GOOGLE_CLOUD_PROJECT)),
           "--prompt"
           ]
    cmd_for_log.extend(str(prompt).split())
    # --- END FIX ---
    
    log_command_to_file(
        output_path.parent, "ai_repair", cmd_for_log, "Repair image with Generative AI"
    )
    
    # Use the robust activation command
    full = conda_prefix_cmd_new(IMAGEN_ENV, cmd_for_exec)
    
    tracker.substep(f"Running Imagen 3 Repair", f"prompt='{prompt[:40]}...'")
    
    # Pass the correct cwd and clean_env
    rc, output = run_cmd(full, cwd=script_cwd, clean_env=True)
    
    # --- FIX: Add proper error raising ---
    if rc != 0:
        # The 'output' now contains the *full* stdout/stderr from the script
        raise RuntimeError(f"Generative AI repair failed.\n\nFull output from script:\n{output}")
    
    # This function should return the path to the *repaired image*
    return output_path


# ... (all other functions, including the broken get_conda_env_python, are removed or unchanged) ...

def run_marigold_cli(image_path: Path, depth_out: Path, marigold_opts: dict, model_path: Path, tracker):
    """
    Run marigold_cli.py to create a 16-bit depth PNG.
    Checks config to see if it should run the NEW Tiled CLI instead.
    """
    
    if not model_path.exists():
        raise RuntimeError(f"Marigold model not found at {model_path}. Please run download_model.py first.")

    tracker.substep("Initializing depth generation")

    tiling_globally_enabled = TILED_DEPTH_SETTINGS.get("enabled", False)
    preset_wants_tiling = marigold_opts.get("use_tiled_depth", False)
    use_tiled = tiling_globally_enabled and preset_wants_tiling

    print(f"Global tiling: {tiling_globally_enabled}")
    print(f"Preset tiling: {preset_wants_tiling}")
    print(f"Result: {use_tiled}")
    pprint(marigold_opts)

    if use_tiled:
        print("Tiling is active.")
        tracker.substep("Using: High-Detail Tiled Depth Pipeline (Slower, Higher Quality)")
        if not MARIGOLD_TILED_CLI.exists():
            raise RuntimeError(f"Tiled CLI not found: {MARIGOLD_TILED_CLI}. Check config.yaml.")
        
        cmd = ["python", "-u", str(MARIGOLD_TILED_CLI),
               "--input", str(image_path),
               "--output", str(depth_out),
               "--checkpoint", str(model_path),
               "--marigold-cli", str(MARIGOLD_CLI), 
               "--global-res", str(TILED_DEPTH_SETTINGS.get("global_res", 768)),
               "--tile-size", str(TILED_DEPTH_SETTINGS.get("tile_size", 1024)),
               "--tile-overlap", str(TILED_DEPTH_SETTINGS.get("tile_overlap", 256)),
               "--tile-steps", str(TILED_DEPTH_SETTINGS.get("tile_steps", 20)),
               "--tile-ensemble", str(TILED_DEPTH_SETTINGS.get("tile_ensemble", 5)),
               "--conda-exe", str(CONDA_EXE),
               "--conda-env", str(MARIGOLD_ENV)
               ]
        
        base_python = "python" if platform.system() == "Windows" else "python3"
        cmd[0] = base_python
        
        # Call with clean_env=False, as the tiled script handles its own env
        rc, output = run_cmd(cmd, cwd=HERE, clean_env=False) 
        
        if rc != 0:
            last_lines = "\n".join(output.splitlines()[-5:])
            raise RuntimeError(f"Tiled depth generation failed.\n\nLast output:\n{last_lines}")
                
    else:
        print("Tiling is unactive.")
        tracker.substep("Using: Standard Depth Pipeline (Faster, Less Detail)")
        tracker.substep(f"Configuration", f"steps={marigold_opts.get('marigold_steps')} ensemble={marigold_opts.get('ensemble')} res={marigold_opts.get('marigold_processing_res')}")
        
        cmd = ["python", "-u", str(MARIGOLD_CLI),
               "--input", str(image_path),
               "--output", str(depth_out),
               "--checkpoint", str(model_path),
               "--steps", str(marigold_opts.get('marigold_steps')),
               "--ensemble", str(marigold_opts.get('marigold_ensemble')),
               "--processing_res", str(marigold_opts.get('marigold_processing_res'))]
        
        full_cmd = conda_prefix_cmd(MARIGOLD_ENV, cmd)
        
        if marigold_opts.get("marigold_match_input_res"):
            full_cmd.append("--match_input_res")
        else:
            full_cmd.append("--no-match_input_res")
        
        # Call with run_cmd, which is correct for this script
        rc, output = run_cmd(full_cmd, cwd=HERE, clean_env=True)
        if rc != 0:
            last_lines = "\n".join(output.splitlines()[-5:])
            raise RuntimeError(f"Marigold depth generation failed.\n\nLast output:\n{last_lines}")
    
    return depth_out


def run_marigold_with_regions(image_path: Path, depth_out: Path, config: dict, tracker):
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
        run_marigold_cli(marigold_input, depth_path, marigold_opts, MARIGOLD_MODEL_PATH, tracker)
    
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
            print(f"   {OK} Using existing: {nobg_path.name}") # <-- Use OK
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
        print(f"      {OK} Processed {region_name}") # <-- Use OK
    
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
    print(f"      {OK} Processed background") # <-- Use OK
    
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
    
    # Set the CWD where the script will run
    script_cwd = EXTRUDE_CLI.parent 
    
    # Calculate all paths RELATIVE to that CWD.
    rel_input_path = os.path.relpath(depth_path, script_cwd)
    rel_output_path = os.path.relpath(stl_out, script_cwd)
    
    # 1. This command is for EXECUTION (relative paths)
    cmd_for_exec = ["python", EXTRUDE_CLI.name,
           "--input", rel_input_path,
           "--output", rel_output_path,
           "--width_mm", str(extrude_params.get("width_mm", 100.0)),
           "--smoothing", str(extrude_params.get("smoothing", 3)),
           "--near_offset", str(extrude_params.get("near_offset", 0.0)),
           "--far_offset", str(extrude_params.get("far_offset", 1.0)),
           "--emboss", str(extrude_params.get("emboss", 0.3)),
           "--f_thic", str(extrude_params.get("f_thic", 0.00)),
           "--f_near", str(extrude_params.get("f_near", -0.0)),
           "--f_back", str(extrude_params.get("f_back", 0.01)),
           "--vertex_colors", str(extrude_params.get("vertex_colors", True)),
           "--scene_lights", str(extrude_params.get("scene_lights", True)),
           "--zip_outputs", str(extrude_params.get("zip_outputs", False))]

    # 2. This command is for LOGGING (absolute paths)
    cmd_for_log = list(cmd_for_exec)
    cmd_for_log[1] = str(EXTRUDE_CLI)  # Use absolute script path
    cmd_for_log[3] = str(depth_path)   # Use absolute input path
    cmd_for_log[5] = str(stl_out)      # Use absolute output path

    # Log command to project file
    if stl_out.parent.exists():
        log_command_to_file(
            stl_out.parent,
            "extrude",
            cmd_for_log, # <-- Use the absolute log command
            f"Convert depth map to 3D model"
        )

    # Get the full conda-wrapped command for execution
    full = conda_prefix_cmd_new(DEPTH_ENV, cmd_for_exec) # <-- Use NEW prefix
    
    print(f"\nConverting depth map to 3D model...")
    print(f"   (Executing in: {script_cwd})") # Debug message
    
    # Pass both cwd and clean_env
    rc, output = run_cmd(full, cwd=script_cwd, clean_env=True)
    
    if rc != 0:
        last_lines = "\n".join(output.splitlines()[-5:])
        raise RuntimeError(f"3D extrusion failed.\n\nLast output from script:\n{last_lines}")
    return stl_out


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
        print("  2. Repair Blurry Photo (AI)") # <-- NEW
        print("  3. Transform 2D to 3D")
        print("  4. Rerun Depth-to-Model (new settings)")
        print("  5. Edit Configuration")
        print("  6. Quit")
        # print("  7. (Debug) Test Aigen Environment") # <-- REMOVED DEBUG OPTION
        print("-"*60)
        
        choice = input("\nSelect option [1-6]: ").strip() # <-- Updated max to 6
        
        if choice == "1":
            generate_ai_image_menu()
        elif choice == "2":
            repair_blurry_photo_menu() # <-- NEW
        elif choice == "3":
            transform_2d_to_3d_menu()
        elif choice == "4":
            rerun_depth_to_model_menu()
        elif choice == "5":
            edit_configuration()
        elif choice == "6":
            print(f"\n{OK} Goodbye!")
            break
        # elif choice == "7": # <-- REMOVED DEBUG OPTION
        #     test_aigen_environment()
        else:
            print(f"\n{ERR} Invalid option. Please choose 1-6.") # <-- Updated max to 6


# --- REMOVED test_aigen_environment() FUNCTION ---


def repair_blurry_photo_menu():
    """Submenu for AI-prompted image repair."""
    
    print(f"\n{'─'*60}")
    print("REPAIR BLURRY PHOTO (AI)")
    print('─'*60)
    print(f"This tool uses Generative AI (Imagen 3) to fix blur,")
    print(f"add detail, and enhance photos before 3D conversion.")
    print(f"{WARN} This is a paid feature (approx. $0.01 per image).")
    print(f"{WARN} Requires 'google_cloud_project' to be set in config.yaml.")
    print('─'*60)
    
    # 1. Select source image
    print("Select source folder:")
    print(f"  1. From {DIR_PHOTOS.name}/ folder")
    print(f"  2. From {DIR_PHOTOS_ENHANCED.name}/ folder")
    print(f"  3. From {DIR_AI.name}/ folder")
    print(f"  4. Back")
    
    source_choice = input("\nSelect folder [1-4]: ").strip()
    
    if source_choice == "1":
        source_dir = DIR_PHOTOS
    elif source_choice == "2":
        source_dir = DIR_PHOTOS_ENHANCED
    elif source_choice == "3":
        source_dir = DIR_AI
    elif source_choice == "4":
        return
    else:
        print(f"\n{ERR} Invalid option.")
        return

    # 2. List files
    files = list_image_files(source_dir)
    if not files:
        print(f"\n{WARN} No images found in {source_dir.name}/")
        input("\nPress Enter to continue...")
        return
    
    print(f"\nAvailable images in {source_dir.name}/:")
    for i, file_path in enumerate(files, 1):
        print(f"  {i}. {file_path.name}")
    print(f"  {len(files) + 1}. Back")
    
    try:
        choice = input(f"\nSelect image to repair [1-{len(files) + 1}]: ").strip()
        choice_num = int(choice)
        
        if choice_num == len(files) + 1:
            return
        
        if 1 <= choice_num <= len(files):
            image_path = files[choice_num - 1]
        else:
            print(f"\n{ERR} Invalid selection.")
            return
    
    except ValueError:
        print(f"\n{ERR} Invalid input.")
        return

    # 3. Get repair prompt
    print(f"\nSelected: {image_path.name}")
    print("Enter a repair prompt (e.g., 'fix blur on face, enhance fur texture'):")
    prompt = input("> ").strip()
    
    if not prompt:
        print(f"\n{ERR} Prompt cannot be empty. Aborting.")
        return

    # 4. Define output path
    output_path = DIR_PHOTOS_ENHANCED / f"{image_path.stem}_AI_repaired.png"
    
    # 5. Run repair
    tracker = TimingTracker(total_steps=1, name="AI Repair")
    try:
        with tracker.step(1, "AI Repair"):
            run_ai_repair_cli(image_path, output_path, prompt, tracker)
        
        tracker.print_summary(output_info=f"Saved: {output_path.name}")
        print(f"\n{OK} AI repair complete!")
        print(f"   You can now find the repaired image in the Photos_enhanced/ folder")
        print(f"   and process it using 'Transform 2D to 3D'.")
        
    except Exception as e:
        print(f"\n{ERR} AI Repair failed: {e}")
    
    input("\nPress Enter to continue...")
# --- END NEW FUNCTION ---


def transform_2d_to_3d_menu():
    """Transform 2D images to 3D models with quality selection."""
    
    print(f"\n{'─'*60}")
    print("TRANSFORM 2D TO 3D")
    print('─'*60)
    print("  1. Ultra-Low Quality (preview, ~30 sec)")
    print("  2. Low Quality (fast, ~2 min)")
    print("  3. Medium Quality (balanced, ~5 min)")
    print("  4. High Quality (best, ~10 min)")
    print("  5. Ultra-High Quality (maximum, ~30 min)")
    print("  6. Batch Process Folder")
    print("  7. Back to main menu")
    print('─'*60)
    
    choice = input("\nSelect option [1-7]: ").strip()
    
    if choice == "1":
        select_and_process("ultra_low_quality")
    elif choice == "2":
        select_and_process("low_quality")
    elif choice == "3":
        select_and_process("medium_quality")
    elif choice == "4":
        select_and_process("high_quality")
    elif choice == "5":
        select_and_process("ultra_high_quality")
    elif choice == "6":
        batch_process_folder(quality="high_quality")
    elif choice == "7":
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
        print(f"\n{WARN} No depth maps found. Generate 3D models first (option 3).")
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
    print(f"  1. From {DIR_AI.name}/ folder")
    print(f"  2. From {DIR_PHOTOS.name}/ folder")
    print(f"  3. From {DIR_PHOTOS_ENHANCED.name}/ folder (pre-enhanced)")
    print(f"  4. Back")
    print('─'*60)
    
    source_choice = input("\nSelect source [1-4]: ").strip()
    
    if source_choice == "1":
        source_dir = DIR_AI
        auto_enhance = False
    elif source_choice == "2":
        source_dir = DIR_PHOTOS
        auto_enhance = cfg.get("auto_enhance_photos", True)
    elif source_choice == "3":
        source_dir = DIR_PHOTOS_ENHANCED
        auto_enhance = False # Already enhanced
    elif source_choice == "4":
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


# --- NEW HELPER FUNCTION ---
def analyze_depth_map(depth_path: Path, tracker: TimingTracker) -> (float, float):
    """
    Analyzes a 16-bit depth map to find the optimal near/far offsets
    by ignoring the masked background.
    """
    tracker.substep("Analyzing 16-bit depth map for 'auto' offsets")
    try:
        img = Image.open(depth_path)
        depth_array = np.array(img).astype(np.uint16)
        
        # This is the "pillar" fix logic. We find the most common
        # pixel value, which is the masked background.
        background_value = st.mode(depth_array, axis=None, keepdims=False).mode
        
        # Create a mask of *only* the subject pixels
        subject_pixels = depth_array[depth_array != background_value]
        
        if subject_pixels.size == 0:
            tracker.substep(f"{WARN} No subject pixels found (image is all one color?). Using defaults.")
            return 0.0, 1.0
            
        subject_min = np.min(subject_pixels)
        subject_max = np.max(subject_pixels)
        
        # Normalize from 0-65535 range to 0.0-1.0 range
        auto_near = subject_min / 65535.0
        auto_far = subject_max / 65535.0
        
        # Add a tiny bit of padding to avoid clipping
        auto_near = max(0.0, auto_near - 0.01)
        auto_far = min(1.0, auto_far + 0.01)
        
        tracker.substep(f"Auto-offsets calculated: near={auto_near:.3f}, far={auto_far:.3f}")
        return auto_near, auto_far
        
    except Exception as e:
        tracker.substep(f"{WARN} analyze_depth_map failed: {e}. Using defaults.")
        return 0.0, 1.0
# --- END NEW FUNCTION ---


def process_single_image(image_path, quality_preset, auto_enhance=False, resume_dir=None):
    """
    Main processing pipeline for single image with comprehensive timing.
    
    Args:
        image_path: Path to source image
        quality_preset: "low_quality", "medium_quality", or "high_quality"
        auto_enhance: Apply photo enhancement before processing
        resume_dir: (Optional) Path object to an existing folder to resume.
    """
    
    # Calculate total steps for timing tracker
    total_steps = 4  # Base: depth generation, extrusion, post-processing, cleanup
    if auto_enhance:
        total_steps += 1
    if REMOVE_BACKGROUND:
        total_steps += 1
    if cfg.get("ai_enhancement", {}).get("enabled", False):
        total_steps += 1
    # Check if wall removal step will run
    if float(EXTRUDE_DEFAULTS.get("f_thic", 0.05)) == 0:
        total_steps += 1
    
    # Initialize timing tracker
    tracker = TimingTracker(
        total_steps=total_steps, 
        name="2D to 3D Pipeline",
        ok_symbol=OK,
        warn_symbol=WARN
    )
    
    try:
        print(f"\n{'='*60}")
        print(f"  Processing: {image_path.name}")
        print(f"  Quality: {quality_preset.replace('_', ' ').title()}")
        if resume_dir:
            print(f"  RESUMING WORK IN EXISTING FOLDER")
        print(f"{'='*60}")
        
        # --- FOLDER CREATION LOGIC ---
        # Always calculate project name from the source file for consistency
        project_name = safe_name_from_file(image_path)

        if resume_dir:
            # Resume mode: Use provided directory exactly
            output_dir = Path(resume_dir).resolve()
            
            if not output_dir.exists():
                 raise RuntimeError(f"Resume directory not found: {output_dir}")
            
            print(f"  {INFO} Resuming in: {output_dir.name}/")
        else:
            # Standard mode: Create NEW unique directory
            output_dir = DIR_3D / f"{project_name}_{quality_preset}"
            
            counter = 2
            while output_dir.exists():
                output_dir = DIR_3D / f"{project_name}_{quality_preset}_{counter}"
                counter += 1
            
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {INFO} Output: {output_dir.name}/\n")
        
        # --- LOG RELAUNCH COMMAND ---
        try:
            pipeline_script_path = (HERE / "2D_to_3D_pipeline.py").resolve()
            
            relaunch_cmd = [
                "python",
                str(pipeline_script_path),
                "--test-mode",
                "--input", str(image_path.resolve()),
                "--quality", re.sub(r'_quality$', '', quality_preset),
                "--resume-work-dir", str(output_dir.resolve()) 
            ]
            
            log_command_to_file(
                output_dir,
                "pipeline_relaunch",
                relaunch_cmd,
                f"Run this command to RESUME processing in this exact folder"
            )
        except Exception as e:
            print(f"  {WARN} Could not log relaunch command: {e}")
        
        working_image = image_path
        step_num = 1
        
        # STEP 1: Photo enhancement (if from photos/ folder)
        if auto_enhance:
            with tracker.step(step_num, "Photo Enhancement"):
                tracker.substep("Checking for cached enhanced version")
                enhanced_path = DIR_PHOTOS_ENHANCED / f"{image_path.stem}_enhanced.png"
                
                if enhanced_path.exists():
                    tracker.substep("Using cached enhanced version")
                    working_image = enhanced_path
                else:
                    if str(SCRIPTS_DIR / "photo_preprocessing") not in sys.path:
                         sys.path.insert(0, str(SCRIPTS_DIR / "photo_preprocessing"))
                    from photo_preprocess import preprocess_photo
                    preset = cfg.get("auto_enhance_preset", "none")
                    tracker.substep(f"Applying '{preset}' enhancement preset")
                    
                    preprocess_photo(
                        str(image_path),
                        str(enhanced_path),
                        preset=preset,
                        save_intermediate=True
                    )
                    working_image = enhanced_path
                    tracker.substep("Enhancement cached for future use")
            step_num += 1
        
        # STEP 2: Background removal
        nobg_path = None
        if REMOVE_BACKGROUND:
            with tracker.step(step_num, "Background Removal"):
                nobg_path = output_dir / f"{project_name}_nobg.png"
                
                f_thick = EXTRUDE_DEFAULTS.get("f_thic", 0.05)
                padding_amount = 0
                if float(f_thick) == 0:
                    padding_amount = 10  # 10px padding
                    tracker.substep(f"INFO: f_thic is 0, will add {padding_amount}px border.")
                
                tracker.substep(f"Removing background ({BG_REMOVAL_METHOD})")
                working_image = remove_background_if_enabled(
                    working_image, 
                    nobg_path, 
                    padding=padding_amount
                )
                tracker.substep("Background removed", "transparent PNG")
            step_num += 1
        
        # Copy source to output
        source_name = image_path.stem
        if not source_name.endswith("_source"):
            source_name += "_source"
        source_save_path = output_dir / f"{source_name}{image_path.suffix}"

        shutil.copy2(image_path, source_save_path)
        
        # STEP 3: AI Enhancement (if enabled)
        ai_config = cfg.get("ai_enhancement", {})
        
        if ai_config.get("enabled", False):
            with tracker.step(step_num, "AI Upscaling & Enhancement"):
                marigold_opts = MARIGOLD_PRESETS[quality_preset]
                target_res = marigold_opts.get("marigold_processing_res", 768)
                
                img = Image.open(working_image)
                current_max = max(img.width, img.height)
                
                ratio = target_res / current_max
                if ratio >= 8: upscale_factor = 8
                elif ratio >= 4: upscale_factor = 4
                elif ratio >= 2: upscale_factor = 2
                else: upscale_factor = 1
                
                tracker.substep(f"Analyzing image size", f"{current_max}px -> target {target_res}px")
                tracker.substep(f"Upscale factor selected", f"{upscale_factor}x")
                
                enhanced_path = output_dir / f"{project_name}_ai_enhanced.png"
                
                ai_enhance_cli = SCRIPTS_DIR / "photo_preprocessing" / "ai_enhance.py"
                script_cwd = ai_enhance_cli.parent
                
                rel_input_path = os.path.relpath(working_image, script_cwd)
                rel_output_path = os.path.relpath(enhanced_path, script_cwd)

                cmd_for_exec = [
                    "python", "-u", ai_enhance_cli.name,
                    "--input", rel_input_path,
                    "--output", rel_output_path,
                    "--upscale", str(upscale_factor),
                    "--method", ai_config.get('upscale_method', 'realesrgan'),
                    "--max-size", str(ai_config.get('max_input_size', 4096)),
                    "--clarity", str(ai_config.get('clarity_strength', 1.3)),
                    "--detail", str(ai_config.get('detail_amount', 1.2)),
                    "--sharpen", str(ai_config.get('sharpen_strength', 150)),
                    "--model-cache-dir", str(GFPGAN_MODELS_DIR.parent)
                ]

                cmd_for_log = list(cmd_for_exec)
                cmd_for_log[2] = str(ai_enhance_cli)
                cmd_for_log[4] = str(working_image)
                cmd_for_log[6] = str(enhanced_path)

                log_command_to_file(
                    output_dir, "ai_enhance", cmd_for_log, "AI upscale and enhance image"
                )
                
                # Use the simple conda prefixer (returns a list)
                # This bypasses cmd.exe quoting issues
                full_cmd = conda_prefix_cmd(PHOTO_PREP_ENV, cmd_for_exec)
                
                tracker.substep("Running AI enhancement pipeline")
                
                try:
                    # Use run_cmd with clean_env=True and cwd set to script dir
                    rc, output = run_cmd(full_cmd, cwd=script_cwd, clean_env=True)
                    
                    if rc != 0:
                        raise RuntimeError(f"AI enhancement failed.\n{output}")
                    
                    working_image = enhanced_path
                    tracker.substep("AI enhancement complete", enhanced_path.name)
                except Exception as e:
                    print(f"  {WARN} AI enhancement failed: {e}")
                    tracker.substep("AI enhancement failed, using original")
            step_num += 1
        else:
            print(f"\n{INFO} AI Enhancement: DISABLED (set ai_enhancement.enabled=true in config)")
        
        # STEP 4: Prepare for Marigold (composite transparent images)
        marigold_input = working_image
        
        if REMOVE_BACKGROUND:
            img = Image.open(working_image) 
            
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                bg_color = cfg.get("marigold_background_color", "white")
                bg_colors = {
                    "gray": (128, 128, 128), "grey": (128, 128, 128),
                    "white": (255, 255, 255), "black": (0, 0, 0),
                    "light_gray": (192, 192, 192), "dark_gray": (64, 64, 64)
                }
                
                if bg_color in bg_colors:
                    bg_rgb = bg_colors[bg_color]
                else:
                    try:
                        if isinstance(bg_color, str) and bg_color.startswith('#'):
                            bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
                        else: bg_rgb = (128, 128, 128)
                    except: bg_rgb = (128, 128, 128)
                
                print(f"  Compositing onto {bg_color} background {bg_rgb}...")
                
                if img.mode != 'RGBA': img = img.convert('RGBA')
                
                background = Image.new('RGB', img.size, bg_rgb)
                background.paste(img, (0, 0), img)
                
                prepared_path = output_dir / f"{project_name}_prepared_for_marigold.png"
                background.save(prepared_path, 'PNG')
                
                log_command_to_file(
                    output_dir,
                    "composite_background",
                    ["# Composited with background color:", str(bg_rgb)],
                    f"Applied {bg_color} background before Marigold"
                )
                marigold_input = prepared_path

        # STEP 5: Generate depth map
        with tracker.step(step_num, "Depth Map Generation"):
            depth_path = output_dir / f"{project_name}_depth_16bit.png"
            marigold_opts = MARIGOLD_PRESETS[quality_preset].copy()
            marigold_opts["quality_preset_key"] = quality_preset 
            
            tracker.substep("Initializing Marigold pipeline")
            
            if cfg.get('region_processing', {}).get('enabled', False):
                run_marigold_with_regions(marigold_input, depth_path, cfg, tracker)
            else:
                run_marigold_cli(marigold_input, depth_path, marigold_opts, MARIGOLD_MODEL_PATH, tracker)
            
            tracker.substep("Depth map generated", f"{depth_path.name}")
            
            # Mask depth map with alpha channel
            if REMOVE_BACKGROUND:
                try:
                    alpha_source_path = working_image 
                    if not alpha_source_path.exists():
                         alpha_source_path = output_dir / f"{project_name}_nobg.png"

                    if alpha_source_path.exists():
                        if str(SCRIPTS_DIR / "photo_preprocessing") not in sys.path:
                             sys.path.insert(0, str(SCRIPTS_DIR / "photo_preprocessing"))
                        from depth_masking import mask_depth_with_alpha
                        tracker.substep(f"Applying alpha mask from {alpha_source_path.name}")
                        mask_depth_with_alpha(depth_path, alpha_source_path)
                        tracker.substep("Depth mask applied (pillar fix)")
                    else:
                        print(f"  {WARN} Could not find alpha source {alpha_source_path.name}, skipping mask.")
                except Exception as e:
                    print(f"  {WARN} Could not mask depth map: {e}")
        step_num += 1
        
        # Extrusion offsets logic
        extrude_settings = EXTRUDE_DEFAULTS.copy()
        auto_near_val = extrude_settings.get("near_offset")
        auto_far_val = extrude_settings.get("far_offset")

        if auto_near_val == "auto" or auto_far_val == "auto":
            auto_near_offset, auto_far_offset = analyze_depth_map(depth_path, tracker)
            if auto_near_val == "auto":
                extrude_settings["near_offset"] = auto_near_offset
                tracker.substep(f"Auto near_offset set to {auto_near_offset:.3f}")
            if auto_far_val == "auto":
                extrude_settings["far_offset"] = auto_far_offset
                tracker.substep(f"Auto far_offset set to {auto_far_offset:.3f}")
        
        # STEP 6: Extrude to 3D model
        with tracker.step(step_num, "3D Model Creation"):
            stl_raw_path = output_dir / f"{project_name}_raw.stl"
            tracker.substep("Converting depth to 3D mesh")
            
            stl_for_repair = stl_raw_path
            run_extrude_cli(depth_path, stl_raw_path, extrude_settings)
            tracker.substep("3D models created", "STL, GLB, OBJ")
        step_num += 1
        
        # STEP 7: Wall Removal (if enabled)
        f_thick = extrude_settings.get("f_thic", 0.05)
        if float(f_thick) == 0:
            with tracker.step(step_num, "Wall Removal"):
                tracker.substep("Removing frame walls (f_thic=0)")
        
                stl_no_walls_path = output_dir / f"{project_name}_no_walls.stl"
                remove_walls_cli = SCRIPTS_DIR / "model_generation" / "remove_walls.py"
                script_cwd = remove_walls_cli.parent

                rel_input_path = os.path.relpath(stl_raw_path, script_cwd)
                rel_output_path = os.path.relpath(stl_no_walls_path, script_cwd)

                cmd_for_exec = [
                    "python", "-u", remove_walls_cli.name,
                    "--input", rel_input_path,
                    "--output", rel_output_path,
                    "--frame-thickness", "0.0"
                ]

                cmd_for_log = list(cmd_for_exec)
                cmd_for_log[2] = str(remove_walls_cli)
                cmd_for_log[4] = str(stl_raw_path)
                cmd_for_log[6] = str(stl_no_walls_path)

                log_command_to_file(
                    output_dir,
                    "remove_walls",
                    cmd_for_log,
                    "Remove 0-thickness walls and add solid bottom"
                )

                full_cmd = conda_prefix_cmd_new(DEPTH_ENV, cmd_for_exec)
                
                try:
                    rc, output = run_cmd(full_cmd, cwd=script_cwd, clean_env=True)
                    if rc != 0:
                        raise RuntimeError(f"Wall removal failed.\n{output}")

                    stl_for_repair = stl_no_walls_path
                    tracker.substep("Frame walls removed", stl_no_walls_path.name)
                except Exception as e:
                    print(f"  {WARN} Wall removal failed: {e}")
                    stl_for_repair = stl_raw_path
            step_num += 1
        
        # STEP 8: Mesh Post-Processing
        with tracker.step(step_num, "Mesh Post-Processing"):
            if str(SCRIPTS_DIR / "model_generation") not in sys.path:
                 sys.path.insert(0, str(SCRIPTS_DIR / "model_generation"))
            from mesh_postprocess import should_repair_for_quality, repair_mesh_via_subprocess
            
            should_repair, repair_settings = should_repair_for_quality(quality_preset, cfg)
            stl_final_path = stl_for_repair # Default
            
            if should_repair:
                tracker.substep(f"Mesh repair enabled for {quality_preset.replace('_', ' ')}")
                stl_final_base_path = output_dir / f"{project_name}" 
                
                tracker.substep("Logging repair commands to file")
                log_command_to_file(
                    output_dir,
                    "mesh_repair_conceptual",
                    ["# See individual commands below"],
                    "Repair and optimize mesh to multiple targets"
                )
                
                try:
                    repaired_files = repair_mesh_via_subprocess(
                        stl_for_repair,
                        stl_final_base_path, 
                        repair_settings,
                        extrude_settings.get("width_mm", 100.0),
                        CONDA_EXE,
                        DEPTH_ENV,
                        conda_prefix_cmd_new
                    )
                    
                    tracker.substep(f"Mesh repair complete, {len(repaired_files)} file(s) generated")
                    
                    if cfg.get("mesh_repair_settings", {}).get("save_before_repair", True):
                        before_repair_path = output_dir / stl_for_repair.name
                        if stl_for_repair != before_repair_path:
                             shutil.copy2(stl_for_repair, before_repair_path)
                        tracker.substep("Pre-repair mesh saved", before_repair_path.name)
                        
                    if repaired_files:
                        main_repaired_file = repaired_files[0]
                        shutil.copy2(main_repaired_file, output_dir / f"{project_name}.stl")
                        tracker.substep("Main file set to", main_repaired_file.name)
                        stl_final_path = output_dir / f"{project_name}.stl"

                except Exception as e:
                    print(f"  {ERR} Mesh repair failed: {e}")
                    tracker.substep("Using unrepaired mesh")
            else:
                tracker.substep(f"Mesh repair disabled for {quality_preset.replace('_', ' ')}")
            
            if stl_final_path.resolve() != (output_dir / f"{project_name}.stl").resolve():
                 shutil.copy2(stl_final_path, output_dir / f"{project_name}.stl")
                 stl_final_path = output_dir / f"{project_name}.stl"

            output_formats = {
                'stl': extrude_settings.get('output_stl', True), 
                'glb': extrude_settings.get('output_glb', False),
                'obj': extrude_settings.get('output_obj', False)
            }
            
            delete_base = stl_raw_path.with_suffix('')
            
            for fmt, keep in output_formats.items():
                final_stl_path_check = output_dir / f"{project_name}.stl"
                check_path = delete_base.with_suffix(f'.{fmt}')
                
                if not keep and check_path.exists() and check_path.resolve() != final_stl_path_check.resolve():
                    check_path.unlink()
                    tracker.substep(f"Removed unwanted format", check_path.name)
        step_num += 1
        
        if cfg.get("delete_source_after_processing", False):
            try:
                image_path.unlink()
                print(f"  {TRASH} Deleted source file")
            except:
                pass
        
        tracker.print_summary(output_info=f"Output: {output_dir.name}/")
        
    except Exception as e:
        print(f"\n{ERR} Processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

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
    print(f"{OK} Loaded latest extrusion settings from config.yaml") # <-- Use OK

    # Find source image
    source_candidates = list(output_dir.glob("source.*"))
    source_image = source_candidates[0] if source_candidates else None
    
    # --- NEW: Check for "auto" settings ---
    tracker = TimingTracker(total_steps=1) # Dummy tracker for analysis
    extrude_settings = EXTRUDE_DEFAULTS.copy()
    auto_near_val = extrude_settings.get("near_offset")
    auto_far_val = extrude_settings.get("far_offset")

    if auto_near_val == "auto" or auto_far_val == "auto":
        auto_near_offset, auto_far_offset = analyze_depth_map(depth_path, tracker)
        if auto_near_val == "auto":
            extrude_settings["near_offset"] = auto_near_offset
        if auto_far_val == "auto":
            extrude_settings["far_offset"] = auto_far_offset
    # --- END NEW ---
    
    # Extrude
    stl_raw_path = output_dir / f"{project_name}_raw_v2.stl"
    run_extrude_cli(depth_path, stl_raw_path, extrude_settings) # <-- Use modified settings
    
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
    print(f"  1. Process all {DIR_AI.name}/")
    print(f"  2. Process all {DIR_PHOTOS.name}/ (with enhancement)")
    print(f"  3. Process all {DIR_PHOTOS_ENHANCED.name}/ (no enhancement)")
    print("  4. Back")
    print('─'*60)
    
    choice = input("\nSelect [1-4]: ").strip()
    
    if choice == "1":
        source_dir = DIR_AI
        auto_enhance = False
        print(f"\n{INFO} Batch processing: {source_dir.relative_to(HERE.parent)}")
    elif choice == "2":
        source_dir = DIR_PHOTOS
        auto_enhance = True
        print(f"\n{INFO} Batch processing: {source_dir.relative_to(HERE.parent)}")
    elif choice == "3":
        source_dir = DIR_PHOTOS_ENHANCED
        auto_enhance = False
        print(f"\n{INFO} Batch processing: {source_dir.relative_to(HERE.parent)}")
    elif choice == "4OpenAI'":
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
        print(f"\n--- BATCH PROGRESS: {i}/{len(files)} ---")
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
    global cfg, EXTRUDE_DEFAULTS, MARIGOLD_PRESETS, GOOGLE_CLOUD_PROJECT
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Re-load globals
    EXTRUDE_DEFAULTS = cfg.get("extrude_defaults", {})
    MARIGOLD_PRESETS = cfg.get("marigold_presets", {})
    GOOGLE_CLOUD_PROJECT = cfg.get("google_cloud_project", "gen-lang-client-0248735054")
    
    print(f"{OK} Configuration reloaded.")


def list_image_files(directory):
    """List all valid image files in directory."""
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = []
    
    if not directory.exists():
        print(f"  {WARN} Directory does not exist: {directory}")
        return files
    
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in valid_extensions:
            files.append(p)
    
    return sorted(files)


# ADD THIS NEW FUNCTION HERE:
def load_prompts():
    """
    Load prompt templates from prompts.json.
    Returns dict with prompt templates and base quality settings.
    """
    prompts_path = HERE / cfg.get("prompts_file", "prompts.json")
    
    if not prompts_path.exists():
        print(f"  {WARN} prompts.json not found, using minimal defaults")
        # Return minimal default structure
        return {
            "base_template": {
                "prefix": "Grayscale, photorealistic, razor-sharp, ultra-detailed, single image suitable for bas relief and CNC cutout, designed to maximize perceived depth and dynamic form.",
                "suffix": "Studio quality, 8K resolution, hyper-sharp focus throughout the entire object with microscopic detail. The image should resemble a highly detailed heightmap, emphasizing volumetric form through subtle tonal variations from pure black to pure white. Perfect edge definition with no soft focus. Optimal lighting to reveal all surface variations and contours.",
                "negative": "multiple views, extreme perspective, heavy distortion, bokeh, atmospheric haze, motion blur, color, sepia, tinted backgrounds, text, watermarks, logos, busy patterns, cluttered composition, soft focus, depth of field, vignetting, chromatic aberration, noise, grain, artifacts, compression, low resolution, blurry edges, flat lighting, overexposure, underexposure, harsh shadows, reflections, reflective, glare, glossy, shiny, flat, 2d, illustration, drawing"
            },
            "prompts": {
                "default": {
                    "name": "Default",
                    "description": "Standard view",
                    "view_description": "{subject}"
                }
            },
            "default_prompt": "default",
            "custom_prompt_template": {
                "name": "Custom",
                "description": "Enter your own prompt"
            }
        }
    
    try:
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  {ERR} Error loading prompts.json: {e}")
        print(f"  {INFO} Using minimal defaults")
        # --- FIX: Return the default structure directly ---
        return {
            "base_template": {
                "prefix": "Grayscale, photorealistic, suitable for bas relief.",
                "suffix": "High quality, detailed.",
                "negative": "color, blur, low quality"
            },
            "prompts": {
                "default": {
                    "name": "Default",
                    "description": "Standard view",
                    "view_description": "{subject}"
                }
            },
            "default_prompt": "default",
            "custom_prompt_template": {
                "name": "Custom",
                "description": "Enter your own prompt"
            }
        }


def build_full_prompt(subject: str, prompt_style: str, prompts_data: dict) -> str:
    """
    Build complete AI generation prompt from subject and style.
    
    Args:
        subject: User's subject description (e.g., "jumping frog")
        prompt_style: Key from prompts.json (e.g., "side_profile") or "custom"
        prompts_data: Loaded prompts.json data
    
    Returns:
        Complete formatted prompt string
    """
    base = prompts_data["base_template"]
    
    if prompt_style == "custom":
        # Custom prompt: user subject + quality wrappers
        return f"{base['prefix']} {subject}. {base['suffix']}"
    
    # Structured prompt: base + view description + subject + suffix
    prompt_config = prompts_data["prompts"].get(prompt_style)
    if not prompt_config:
        # Fallback to custom if style not found
        return f"{base['prefix']} {subject}. {base['suffix']}"
    
    view_desc = prompt_config["view_description"].format(subject=subject)
    
    full_prompt = f"{base['prefix']} {view_desc} {base['suffix']}"
    
    return full_prompt


def run_automated(args):
    """
    Run pipeline in automated test mode with CLI arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    # Validate required args
    if not args.input:
        print(f"{ERR} --input is required in test mode")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"{ERR} Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine quality preset
    if args.quality:
        quality_key = f"{args.quality}_quality"
    else:
        quality_key = "medium_quality"  # Default
    
    if quality_key not in MARIGOLD_PRESETS:
        print(f"{ERR} Unknown quality preset: {args.quality}")
        print(f"Available presets: {', '.join(MARIGOLD_PRESETS.keys())}")
        sys.exit(1)
    
    # Get base configurations
    marigold_preset = MARIGOLD_PRESETS[quality_key].copy()
    extrude_settings = EXTRUDE_DEFAULTS.copy()
    
    # Apply CLI overrides
    marigold_preset, extrude_settings = apply_cli_overrides(
        marigold_preset, extrude_settings, args
    )
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"  AUTOMATED TEST MODE")
    print(f"{'='*60}")
    print(f"  Input: {input_path.name}")
    print(f"  Base preset: {args.quality or 'medium'}")
    
    # Show overrides if any
    overrides = []
    if args.steps: overrides.append(f"steps={args.steps}")
    if args.ensemble: overrides.append(f"ensemble={args.ensemble}")
    if args.resolution: overrides.append(f"resolution={args.resolution}")
    if args.emboss: overrides.append(f"emboss={args.emboss}")
    if args.smoothing: overrides.append(f"smoothing={args.smoothing}")
    if args.near is not None: overrides.append(f"near={args.near}")
    if args.far is not None: overrides.append(f"far={args.far}")
    if args.width: overrides.append(f"width={args.width}")
    
    if overrides:
        print(f"  Overrides: {', '.join(overrides)}")
    
    print(f"\n  Final Configuration:")
    print(f"    Marigold: steps={marigold_preset.get('marigold_steps')} " +
          f"ensemble={marigold_preset.get('marigold_ensemble')} " +
          f"res={marigold_preset.get('marigold_processing_res')}")
    print(f"    Extrusion: emboss={extrude_settings.get('emboss')} " +
          f"smoothing={extrude_settings.get('smoothing')} " +
          f"near/far={extrude_settings.get('near_offset')}/{extrude_settings.get('far_offset')}")
    print(f"{'='*60}")
    
    # Determine auto_enhance based on source
    auto_enhance = False
    if input_path.parent == DIR_PHOTOS:
        auto_enhance = cfg.get("auto_enhance_photos", True)
    
    # Override if CLI flag set
    if args.no_enhancement:
        auto_enhance = False
    
    # Temporarily override REMOVE_BACKGROUND if flag set
    global REMOVE_BACKGROUND
    original_bg_setting = REMOVE_BACKGROUND
    if args.no_bg_removal:
        REMOVE_BACKGROUND = False
    
    try:
        # Update the global presets for this run
        MARIGOLD_PRESETS[quality_key] = marigold_preset
        EXTRUDE_DEFAULTS.update(extrude_settings)
        
        # Run processing
        process_single_image(
            input_path, 
            quality_key, 
            auto_enhance, 
            resume_dir=args.resume_work_dir
        )
    finally:
        # Restore original setting
        REMOVE_BACKGROUND = original_bg_setting


if __name__ == "__main__":
    try:
        args = parse_cli_args()
        
        if args.test_mode:
            run_automated(args)
        else:
            main_menu()
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")