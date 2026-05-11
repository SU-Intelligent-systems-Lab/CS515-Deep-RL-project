"""PyTorch device helpers."""
import numpy as np
import torch


device: torch.device = torch.device("cpu")


def init_gpu(use_gpu: bool = True, gpu_id: int = 0) -> None:
    """Set the module-global `device` to CUDA if available, else CPU."""
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU id {gpu_id}")
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def from_numpy(array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """numpy array -> tensor on `device`."""
    return torch.from_numpy(array).to(dtype=dtype, device=device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """tensor -> detached numpy on CPU."""
    return tensor.detach().to("cpu").numpy()
