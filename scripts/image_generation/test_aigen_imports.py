#!/usr/bin/env python3
"""
Simple test script to check 'aigen' environment imports.
This script is called by the main pipeline's debug menu.
"""

import sys
import os

print("[TEST] --- Aigen Import Test Starting ---")

try:
    print(f"[TEST] Python Executable: {sys.executable}")
    print(f"[TEST] Python Version: {sys.version.split()[0]}")
    
    # Test 1: Pillow
    print("[TEST] Importing 'PIL' (Pillow)...")
    from PIL import Image
    print(f"[TEST] ... Pillow OK. Version: {Image.__version__}")
    
    # Test 2: Google Auth
    print("[TEST] Importing 'google.auth'...")
    import google.auth
    print(f"[TEST] ... google.auth OK.")
    
    # Test 3: Vertex AI (Base)
    print("[TEST] Importing 'vertexai'...")
    import vertexai
    print(f"[TEST] ... vertexai OK.")
    
    # Test 4: Vertex AI Vision Models (The failing import)
    print("[TEST] Importing 'vertexai.vision_models'...")
    
    # --- THIS IS THE FIX ---
    # We are removing '.preview' because you have a modern version
    # of the library where these classes are no longer in preview.
    from vertexai.vision_models import ImageGenerationModel, Image, EditImageParams
    # --- END FIX ---
    
    print(f"[TEST] ... vertexai.vision_models OK.")
    print(f"[TEST]     Successfully imported:")
    print(f"[TEST]     - ImageGenerationModel")
    print(f"[TEST]     - Image")
    print(f"[TEST]     - EditImageParams")
    
    print("\n[TEST] --- ALL IMPORTS PASSED ---")
    sys.exit(0) # Exit with success code

except ImportError as e:
    print("\n[X] TEST FAILED: Could not import a library.", file=sys.stderr)
    print(f"[X] ERROR: {e}", file=sys.stderr)
    sys.exit(1) # Exit with error code

except Exception as e:
    print(f"\n[X] TEST FAILED: An unexpected error occurred.", file=sys.stderr)
    print(f"[X] ERROR: {e}", file=sys.stderr)
    sys.exit(1)