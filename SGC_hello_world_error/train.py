"""
Hello world SGC job that intentionally errors during init.
Used to test SGC error handling and run status behavior.
"""

import torch

print(">>> Starting init...")
print(f"CUDA available: {torch.cuda.is_available()}")

# Intentional error during init
raise RuntimeError(
    "Intentional init error for testing SGC error handling. "
    "This simulates a misconfigured job that fails before training starts."
)
