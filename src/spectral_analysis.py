"""
Spectral analysis module: applies FFT and extracts dominant frequency components.
"""

import numpy as np


def compute_fft(prices: np.ndarray, sampling_period: float = 1.0):
    """
    Compute the one-sided FFT of a price series.

    Args:
        prices: 1-D array of closing prices.
        sampling_period: Time between samples in hours (default 1h).

    Returns:
        frequencies: Array of frequency values (cycles per hour).
        amplitudes: Corresponding amplitude for each frequency.
        fft_values: Raw complex FFT coefficients (full spectrum).
    """
    n = len(prices)
    fft_values = np.fft.fft(prices)

    # One-sided spectrum (positive frequencies only)
    frequencies = np.fft.fftfreq(n, d=sampling_period)
    amplitudes = (2.0 / n) * np.abs(fft_values)

    # Keep only the positive-frequency half
    pos_mask = frequencies > 0
    return frequencies[pos_mask], amplitudes[pos_mask], fft_values


def find_dominant_frequencies(frequencies: np.ndarray, amplitudes: np.ndarray, top_n: int = 5):
    """
    Identify the top-N dominant frequencies by amplitude.

    Args:
        frequencies: Positive-frequency array.
        amplitudes: Corresponding amplitudes.
        top_n: How many dominant components to return.

    Returns:
        dominant_freqs: Frequencies of the strongest components.
        dominant_amps: Their amplitudes.
        dominant_indices: Indices into the input arrays.
    """
    top_n = min(top_n, len(amplitudes))
    indices = np.argsort(amplitudes)[::-1][:top_n]
    return frequencies[indices], amplitudes[indices], indices
