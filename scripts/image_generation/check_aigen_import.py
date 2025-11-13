import sys
import vertexai
from vertexai.vision_models import ImageGenerationModel

print("--- AIgen Import Test ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"VertexAI Version: {vertexai.__version__}")
print("-" * 30)

# 1. Check the main vision_models module
try:
    import vertexai.vision_models
    print("\n[SUCCESS] Found 'vertexai.vision_models'")
    
    # Get all attributes
    all_attrs = dir(vertexai.vision_models)
    
    # Filter for parameter-related classes
    param_classes = [name for name in all_attrs if "Param" in name or "Config" in name]
    
    if param_classes:
        print("\nFound Parameter/Config classes:")
        for name in param_classes:
            print(f"  - {name}")
    else:
        print("\n[INFO] No 'Param' or 'Config' classes found in main module.")
        
except ImportError as e:
    print(f"\n[FAIL] Could not import 'vertexai.vision_models': {e}")

print("-" * 30)

# 2. Check the preview.vision_models module
try:
    import vertexai.preview.vision_models
    print("\n[SUCCESS] Found 'vertexai.preview.vision_models'")
    
    # Get all attributes
    all_attrs_preview = dir(vertexai.preview.vision_models)
    
    # Filter for parameter-related classes
    param_classes_preview = [name for name in all_attrs_preview if "Param" in name or "Config" in name]
    
    if param_classes_preview:
        print("\nFound Parameter/Config classes in PREVIEW:")
        for name in param_classes_preview:
            print(f"  - {name}")
    else:
        print("\n[INFO] No 'Param' or 'Config' classes found in preview module.")

except ImportError as e:
    print(f"\n[FAIL] Could not import 'vertexai.preview.vision_models': {e}")

print("-" * 30)

# 3. Check the attributes of the model's edit_image method
try:
    print("\nChecking 'edit_image' method parameters...")
    # Get a reference to the method without calling it
    method_info = ImageGenerationModel.edit_image
    
    # Print its signature (if available)
    import inspect
    print(f"  Signature: {inspect.signature(method_info)}")
    
except Exception as e:
    print(f"\n[FAIL] Could not inspect 'edit_image' method: {e}")

print("\n--- Test Complete ---")