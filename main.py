"""
Crypto Market Spectral Decomposition Engine
============================================
Pipeline: Fetch Data -> Clean -> FFT -> Analyze -> Filter -> Reconstruct -> Plot

Modes:
  "2d" — classic FFT decomposition with matplotlib plots
  "3d" — STFT time-frequency analysis with interactive Plotly 3D surface
"""

from src.data_fetcher import fetch_ohlcv
from src.spectral_analysis import compute_fft, find_dominant_frequencies
from src.signal_filter import filter_signal, compute_residual
from src.visualizer import plot_all
from src.advanced_visualizer import compute_stft, plot_3d_surface

# --- Configuration ---
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 500          # number of candles
TOP_N = 5            # dominant frequency components to keep
SAMPLING_PERIOD = 1  # 1 hour between samples
MODE = "3d"          # "2d" or "3d"


def run_2d(df, prices):
    """Classic FFT decomposition with 2D matplotlib plots."""
    frequencies, amplitudes, fft_values = compute_fft(prices, sampling_period=SAMPLING_PERIOD)

    dom_freqs, dom_amps, _ = find_dominant_frequencies(frequencies, amplitudes, top_n=TOP_N)
    print(f"\nTop {TOP_N} dominant cycles:")
    for freq, amp in zip(dom_freqs, dom_amps):
        period_hours = 1.0 / freq
        print(f"  Period: {period_hours:8.1f} hours  |  Amplitude: {amp:10.2f}")

    print("\nFiltering noise (keeping top components)...")
    filtered = filter_signal(fft_values, top_n=TOP_N)
    residual = compute_residual(prices, filtered)

    print("Generating 2D plots...")
    plot_all(
        timestamps=df.index,
        original=prices,
        filtered=filtered,
        residual=residual,
        frequencies=frequencies,
        amplitudes=amplitudes,
        symbol=SYMBOL,
    )


def run_3d(prices):
    """STFT time-frequency analysis with interactive 3D Plotly surface."""
    print("Computing Short-Time Fourier Transform (STFT)...")
    freqs, times, amplitude = compute_stft(prices, fs=1.0 / SAMPLING_PERIOD)

    print(f"  STFT matrix: {amplitude.shape[0]} freq bins x {amplitude.shape[1]} time segments")
    print("Generating interactive 3D surface...")
    plot_3d_surface(freqs, times, amplitude, symbol=SYMBOL)


def main():
    # 1. Fetch data
    print(f"Fetching {LIMIT} candles of {SYMBOL} ({TIMEFRAME})...")
    df = fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    print(f"Retrieved {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    prices = df["close"].values

    # 2. Run selected mode
    if MODE == "3d":
        run_3d(prices)
    else:
        run_2d(df, prices)

    print("Done.")


if __name__ == "__main__":
    main()
