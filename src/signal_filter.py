"""
Signal filter module: removes noise by keeping only dominant frequency components,
then reconstructs the cleaned signal via inverse FFT.
"""

import numpy as np


def filter_signal(fft_values: np.ndarray, top_n: int = 5) -> np.ndarray:
    """
    Zero out all but the top-N frequency components (by magnitude) and
    reconstruct the signal with inverse FFT.

    The DC component (index 0) is always preserved so the reconstructed
    signal sits at the correct price level.

    Args:
        fft_values: Full complex FFT array from np.fft.fft.
        top_n: Number of dominant frequency components to keep.

    Returns:
        Reconstructed (filtered) signal as a real-valued array.
    """
    n = len(fft_values)
    magnitudes = np.abs(fft_values).copy()

    # Exclude DC (index 0) from ranking so it is always kept
    magnitudes[0] = 0.0

    # Find indices of the top-N strongest components
    top_indices = np.argsort(magnitudes)[::-1][:top_n]

    # Build a mask: keep DC + top-N
    filtered = np.zeros_like(fft_values)
    filtered[0] = fft_values[0]  # DC component
    for idx in top_indices:
        filtered[idx] = fft_values[idx]

    # Inverse FFT to get the time-domain signal back
    return np.real(np.fft.ifft(filtered))


def compute_residual(original: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    """
    Compute the residual (noise) between original and filtered signals.
    """
    return original - filtered
