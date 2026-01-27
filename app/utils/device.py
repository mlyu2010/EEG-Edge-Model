"""
Device detection and management utilities.
"""
import torch
from loguru import logger


def get_device(preference: str = "auto") -> str:
    """
    Get the best available device for PyTorch.

    Args:
        preference: Device preference - 'auto', 'cpu', 'cuda', or 'mps'

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if preference == "cpu":
        logger.info("Using CPU (forced by preference)")
        return "cpu"

    if preference == "cuda":
        if torch.cuda.is_available():
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"

    if preference == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using Apple Silicon GPU (MPS)")
            return "mps"
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"

    # Auto-detect best device
    if torch.cuda.is_available():
        logger.info(f"Auto-detected CUDA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"

    if torch.backends.mps.is_available():
        logger.info("Auto-detected Apple Silicon GPU (MPS)")
        return "mps"

    logger.info("Using CPU (no GPU acceleration available)")
    return "cpu"


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "cpu": True,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    if info["mps_available"]:
        info["mps_built"] = torch.backends.mps.is_built()

    return info


def print_device_info():
    """Print information about available devices."""
    info = get_device_info()

    print("\n" + "=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    print(f"CPU: Available")
    print(f"CUDA: {'Available' if info['cuda_available'] else 'Not available'}")

    if info["cuda_available"]:
        print(f"  - Device count: {info['cuda_device_count']}")
        print(f"  - Device name: {info['cuda_device_name']}")
        print(f"  - CUDA version: {info['cuda_version']}")

    print(f"MPS (Apple Silicon): {'Available' if info['mps_available'] else 'Not available'}")

    if info["mps_available"]:
        print(f"  - Built: {info['mps_built']}")

    print("=" * 60 + "\n")
