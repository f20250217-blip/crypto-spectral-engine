"""
Advanced visualizer module: STFT on log returns for balanced
time-frequency analysis with clean 3D Plotly surface.
"""

import os
import numpy as np
import plotly.graph_objects as go
from scipy.signal import stft
from scipy.ndimage import gaussian_filter


def compute_stft(prices: np.ndarray, fs: float = 1.0):
    """
    Compute STFT on normalised log returns (not raw prices).

    Log returns remove price-level bias and produce a stationary signal
    whose frequency content reflects actual periodic behaviour.

    Args:
        prices: 1-D array of closing prices.
        fs: Sampling frequency (samples per hour, default 1).

    Returns:
        freqs: Frequency bins (cycles / hour).
        times: Time-segment centres (hours from start).
        power: 2-D matrix (freq x time), normalised log-power spectrum.
    """
    # Step 1: Log returns — stationary, removes price-level trend
    returns = np.diff(np.log(prices))

    # Step 2: Normalise to zero-mean, unit-variance
    returns = (returns - np.mean(returns)) / np.std(returns)

    # Step 3: STFT with zero-padded FFT for frequency interpolation
    nperseg = min(64, len(returns))
    noverlap = min(48, nperseg - 1)

    freqs, times, Zxx = stft(
        returns,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=128,
        boundary=None,
    )

    # Step 4: Power spectrum
    power = np.abs(Zxx) ** 2

    # Step 5: Log scale
    power = np.log1p(power)

    # Step 6: Normalise to [0, 1]
    pmax = np.max(power)
    if pmax > 0:
        power = power / pmax

    # Step 7: Light smoothing — preserve ridge structure
    power = gaussian_filter(power, sigma=0.8)

    return freqs, times, power


def plot_3d_surface(
    freqs: np.ndarray,
    times: np.ndarray,
    power: np.ndarray,
    symbol: str = "BTC/USDT",
):
    """
    Render a clean 3D surface of the STFT power spectrogram.

    Exports:
        spectral_3d_surface.html  — interactive (opens in browser)
        output/3d_preview.png     — static preview for GitHub
    """
    # Percentile scaling for better mid-level contrast
    cmin = float(np.percentile(power, 5))
    cmax = float(np.percentile(power, 95))

    fig = go.Figure(
        data=[
            go.Surface(
                x=times,
                y=freqs,
                z=power,
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Power",
                        side="right",
                        font=dict(size=13, color="#CCCCCC"),
                    ),
                    thickness=14,
                    len=0.5,
                    tickfont=dict(size=11, color="#999999"),
                    outlinewidth=0,
                ),
                opacity=0.95,
            )
        ]
    )

    axis_common = dict(
        color="#999999",
        gridcolor="rgba(255,255,255,0.04)",
        showbackground=True,
        backgroundcolor="#121620",
        tickfont=dict(size=11, color="#888888"),
    )

    fig.update_layout(
        title=dict(
            text=f"{symbol} Spectral Surface (STFT on Log Returns)",
            font=dict(size=20, color="#DDDDDD", family="Arial, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title=dict(text="Time (hours)", font=dict(size=13)), nticks=8, **axis_common),
            yaxis=dict(title=dict(text="Frequency (cycles/hour)", font=dict(size=13)), nticks=6, **axis_common),
            zaxis=dict(title=dict(text="Power", font=dict(size=13)), nticks=5, **axis_common),
            bgcolor="#0f1116",
            aspectratio=dict(x=1.5, y=1.0, z=0.7),
            camera=dict(eye=dict(x=1.6, y=1.3, z=0.9)),
        ),
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="#CCCCCC", family="Arial, sans-serif"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # --- Interactive HTML ---
    html_file = "output/spectral_3d_surface.html"
    fig.write_html(html_file, auto_open=True)
    print(f"Interactive 3D plot saved to {html_file}")

    # --- Static PNG preview ---
    os.makedirs("output", exist_ok=True)
    png_file = "output/3d_preview.png"
    try:
        fig.write_image(png_file, width=2200, height=1400, scale=3)
        print(f"Static preview saved to {png_file}")
    except Exception:
        print(f"Note: install 'kaleido' (pip install kaleido) to export {png_file}")
