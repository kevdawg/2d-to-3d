#!/usr/bin/env python3
"""
Mesh Repair and Optimization
Wrapper that calls mesh_repair_cli.py via subprocess in correct conda environment.
Handles multiple target outputs (face count and density).
"""
from pathlib import Path
import subprocess
import sys
import yaml 
import os 

# Define the console prefixes
OK = "[OK]"
ERR = "[X]"
WARN = "[!]"
INFO = "[i]"

def should_repair_for_quality(quality_preset: str, config: dict) -> tuple:
    # ... (this function is unchanged) ...
    repair_mode = config.get("enable_mesh_repair", "auto")
    
    if repair_mode is True:
        return True, config.get("mesh_repair_settings", {})
    
    if repair_mode is False:
        return False, {}
    
    if repair_mode == "auto":
        quality_rules = config.get("mesh_repair_by_quality", {})
        preset_rules = quality_rules.get(quality_preset, {})
        
        if not preset_rules.get("enabled", False):
            return False, {}
        
        default_settings = config.get("mesh_repair_settings", {})
        merged_settings = {**default_settings, **preset_rules}
        merged_settings.pop('enabled', None)
        
        return True, merged_settings
    
    return False, {}


def repair_mesh_via_subprocess(
    input_stl: Path, 
    output_base_path: Path, 
    settings: dict, 
    extrude_width_mm: float, 
    conda_exe: str, 
    depth_env: str,
    conda_prefix_cmd_new
    ) -> list:
    """
    Call mesh_repair_cli.py via subprocess for each target output.
    """
    # Get path to CLI script
    script_dir = Path(__file__).parent
    cli_script = script_dir / "mesh_repair_cli.py"
    
    if not cli_script.exists():
        raise RuntimeError(f"mesh_repair_cli.py not found at {cli_script}")

    target_outputs = settings.get('target_outputs', [])
    if not isinstance(target_outputs, list):
        target_outputs = [target_outputs]

    generated_files = []
    
    # Define the Current Working Directory from which the subprocess will run
    run_cwd = cli_script.parent 
    
    # Calculate the relative path for the *input* file once
    rel_input_path = os.path.relpath(input_stl, run_cwd)

    print(f"\n💻 Generating {len(target_outputs)} mesh output(s)...")

    for target in target_outputs:
        
        # Create unique output path based on target
        if isinstance(target, str):
            suffix = f"{target.split('/')[0]}d" 
        else:
            if target >= 1_000_000:
                suffix = f"{target // 1_000_000}m"
            elif target >= 1_000:
                suffix = f"{target // 1_000}k"
            else:
                suffix = f"{target}"
        
        # This is the full, absolute path to the output file
        output_stl_abs = output_base_path.with_name(f"{output_base_path.stem}_{suffix}.stl")
        
        # Calculate the relative path for the *output* file
        rel_output_path = os.path.relpath(output_stl_abs, run_cwd)
        
        # Build command
        cmd_for_exec = [
            "python", "-u", cli_script.name,
            "--input", str(rel_input_path),
            "--output", str(rel_output_path),
            "--target", str(target), 
            "--width-mm", str(extrude_width_mm),
            "--smooth", str(settings.get('smooth_iterations', 0))
        ]
        
        if not settings.get('fill_holes', True):
            cmd_for_exec.append("--no-fill-holes")
        
        if not settings.get('ensure_manifold', True):
            cmd_for_exec.append("--no-manifold")
        
        full_cmd = conda_prefix_cmd_new(depth_env, cmd_for_exec)
        
        print(f"\n  Running repair for target: {target}...")
        
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8', # <-- This is important
                errors='replace',  # <-- This helps handle bad chars
                cwd=run_cwd, 
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"    {line}")
            
            if result.returncode != 0:
                error_output = result.stderr if result.stderr else "Unknown error"
                # Use the 'ERR' variable we defined at the top
                print(f"  {ERR} Mesh repair failed for target {target} (exit code {result.returncode}): {error_output}")
            else:
                generated_files.append(output_stl_abs) # Append the absolute path
            
        except Exception as e:
            # Use the 'ERR' variable we defined at the top
            print(f"  {ERR} Mesh repair subprocess failed for target {target}: {e}")
            
    return generated_files