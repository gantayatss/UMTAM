# test_setup.py
"""
Verify your Mac environment is ready for experiments
"""
import torch
import transformers
import datasets
import numpy as np
import matplotlib

print("=== Environment Check ===\n")

# 1. PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# 2. Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✅ Using device: MPS (GPU acceleration)")
else:
    device = torch.device("cpu")
    print(f"⚠️  Using device: CPU (slower, but will work)")

# 3. Memory test
try:
    # Test creating a tensor on MPS
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = x @ y
    print(f"✅ Memory test passed (allocated 1000x1000 tensors)")
except Exception as e:
    print(f"❌ Memory test failed: {e}")

# 4. Hugging Face
print(f"\nTransformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")

# 5. Quick data loading test
print("\nTesting data loading...")
try:
    dataset = datasets.load_dataset("glue", "sst2", split="validation[:10]")
    print(f"✅ Successfully loaded 10 SST-2 samples")
    print(f"   Sample: {dataset[0]['sentence'][:50]}...")
except Exception as e:
    print(f"❌ Data loading failed: {e}")

print("\n=== Setup Complete ===")
print("You're ready to run experiments!")