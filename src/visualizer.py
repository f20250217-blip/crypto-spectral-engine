"""
Visualizer module: generates 2D plots for spectral decomposition results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_all(
    timestamps: pd.DatetimeIndex,
    original: np.ndarray,
    filtered: np.ndarray,
    residual: np.ndarray,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    symbol: str = "BTC/USDT",
):
    """
    Create three subplots:
      A. Original price vs filtered (reconstructed) signal
      B. Frequency spectrum (frequency vs amplitude)
      C. Residual noise
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(f"{symbol} Spectral Decomposition", fontsize=16, fontweight="bold")

    # --- A. Price vs Filtered Signal ---
    ax = axes[0]
    ax.plot(timestamps, original, linewidth=0.8, label="Original Price", alpha=0.7)
    ax.plot(timestamps, filtered, linewidth=1.5, label="Filtered Signal", color="crimson")
    ax.set_title("Price vs Filtered Signal")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=30)

    # --- B. Frequency Spectrum ---
    ax = axes[1]
    # Convert frequency (cycles/hour) to period (hours) for readability
    ax.stem(frequencies, amplitudes, linefmt="steelblue", markerfmt=" ", basefmt="grey")
    ax.set_title("Frequency Spectrum")
    ax.set_xlabel("Frequency (cycles / hour)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # --- C. Residual Noise ---
    ax = axes[2]
    ax.plot(timestamps, residual, linewidth=0.6, color="grey", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Residual Noise (Original - Filtered)")
    ax.set_ylabel("Residual (USDT)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=30)

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/spectral_decomposition.png", dpi=150)
    print("Plot saved to output/spectral_decomposition.png")
    plt.show()
