import numpy as np

def rescale_minmax(image: np.ndarray) -> np.ndarray:
    """Rescales the grayscale image so that min is 0 and max is 255."""
    min_val = image.min()
    max_val = image.max()
    if max_val == min_val:  # Avoid division by zero
        return np.full_like(image, 127, dtype=np.uint8)  # Neutral gray
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def normalize_and_rescale(image: np.ndarray) -> np.ndarray:
    """Normalizes image to mean 127.5 and std 1.0, then rescales it to [0, 255]."""
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:  # Avoid division by zero
        normalized = np.full_like(image, 127.5)
    else:
        normalized = (image - mean) / std + 127.5
    return rescale_minmax(normalized)