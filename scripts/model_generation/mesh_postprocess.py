#!/usr/bin/env python3
"""
Mesh Repair and Optimization
Wrapper that calls mesh_repair_cli.py via subprocess in correct conda environment.
"""
from pathlib import Path
import subprocess
import sys


def should_repair_for_quality(quality_preset: str, config: dict) -> tuple:
    """
    Determine if mesh repair should run based on quality preset.
    
    Args:
        quality_preset: "low_quality", "medium_quality", or "high_quality"
        config: Full config dict
    
    Returns:
        (should_repair: bool, settings: dict)
    """
    repair_mode = config.get("enable_mesh_repair", "auto")
    
    # Always repair
    if repair_mode is True:
        return True, config.get("mesh_repair_settings", {})
    
    # Never repair
    if repair_mode is False:
        return False, {}
    
    # Auto mode - check quality-specific rules
    if repair_mode == "auto":
        quality_rules = config.get("mesh_repair_by_quality", {})
        preset_rules = quality_rules.get(quality_preset, {})
        
        if not preset_rules.get("enabled", False):
            return False, {}
        
        # Merge preset-specific settings with defaults
        default_settings = config.get("mesh_repair_settings", {})
        merged_settings = {**default_settings, **preset_rules}
        merged_settings.pop('enabled', None)  # Remove 'enabled' key
        
        return True, merged_settings
    
    # Fallback
    return False, {}


def repair_mesh_via_subprocess(input_stl, output_stl, settings, conda_exe, depth_env):
    """
    Call mesh_repair_cli.py via subprocess in depth-to-3d environment.
    
    Args:
        input_stl: Path to raw STL
        output_stl: Path for repaired STL
        settings: Repair settings dict
        conda_exe: Path to conda executable
        depth_env: Name of depth-to-3d environment
    
    Returns:
        Path to repaired mesh
    
    Raises:
        RuntimeError: If repair fails
    """
    # Get path to CLI script
    script_dir = Path(__file__).parent
    cli_script = script_dir / "mesh_repair_cli.py"
    
    if not cli_script.exists():
        raise RuntimeError(f"mesh_repair_cli.py not found at {cli_script}")
    
    # Build command
    cmd = [
        conda_exe, "run", "-n", depth_env, "--no-capture-output",
        "python", str(cli_script),
        "--input", str(input_stl),
        "--output", str(output_stl),
        "--smooth", str(settings.get('smooth_iterations', 0)),
        "--target-faces", str(settings.get('target_faces', 0))
    ]
    
    if not settings.get('fill_holes', True):
        cmd.append("--no-fill-holes")
    
    if not settings.get('ensure_manifold', True):
        cmd.append("--no-manifold")
    
    # Show command for debugging
    print(f"\n💻 Mesh repair command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # Run subprocess
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        # Print output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"    {line}")
        
        if result.returncode != 0:
            error_output = result.stderr if result.stderr else "Unknown error"
            raise RuntimeError(f"Mesh repair failed (exit code {result.returncode}): {error_output}")
        
        return Path(output_stl)
        
    except Exception as e:
        raise RuntimeError(f"Mesh repair subprocess failed: {e}")