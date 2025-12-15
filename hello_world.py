import os
import torch

print("This is a user script!")
print("PyTorch version:", torch.__version__)
# Print an environment variable unique to PyTorch Docker images
print("PYTORCH_VERSION:", os.environ.get("PYTORCH_VERSION"))

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

# simple tensor test
x = torch.tensor([1, 2, 3], device="cuda" if torch.cuda.is_available() else "cpu")
print("Tensor on device:", x.device)
print("Tensor value:", x)
