from typing import Any
import numpy as np 
import torch 

def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
def recursive_to(x: Any, device: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, np.ndarray):
        return to_tensor(x).to(device)
    elif isinstance(x, list):
        return [recursive_to(i, device) for i in x]
    else:
        return x 