import torch
import math
import os

def format_size(size_bytes):
    """Convert bytes to human readable string"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def count_parameters(state_dict):
    """Count parameters in state dictionary recursively"""
    total = 0
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            params = value.numel()
            total += params
            print(f"{key}: {params:,} parameters")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            total += count_parameters(value)
    return total

# Load the model state dictionary
model_dict = torch.load('checkpoints/best_model.pt', map_location='cpu')

print("\nLayer-wise parameter count:")
print("-" * 50)
total_params = count_parameters(model_dict)

# Calculate approximate size (4 bytes per parameter)
total_size = total_params * 4

print("\nSummary:")
print("-" * 50)
print(f"Total parameters: {total_params:,}")
print(f"Approximate model size: {format_size(total_size)}")
print(f"Actual file size: {format_size(os.path.getsize('checkpoints/best_model.pt'))}")