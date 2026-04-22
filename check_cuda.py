import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n--- DETAILED DIAGNOSTIC ---")
    try:
        import ctypes
        # Try to load the CUDA runtime library manually
        ctypes.WinDLL("cudart64_12.dll")
        print("Successfully loaded cudart64_12.dll")
    except Exception as e:
        print(f"Failed to load cudart64_12.dll: {e}")

    try:
        # Check if the driver version is visible to torch
        print(f"CUDA Driver version: {torch.cuda.get_device_capability(0) if torch.cuda.device_count() > 0 else 'No Device'}")
    except Exception as e:
        print(f"Error checking device capability: {e}")
else:
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
