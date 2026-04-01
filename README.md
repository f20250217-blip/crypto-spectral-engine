# Crypto Market Spectral Decomposition Engine

> Extracting hidden market cycles using signal processing and spectral analysis

![3D Spectral Surface](output/3d_preview.png)

This project transforms crypto price data into the frequency domain to reveal hidden cyclical structures and evolving market dynamics.

## Overview

Live BTC/USDT data is fetched from Binance and decomposed using FFT and STFT. The system identifies dominant market cycles, filters noise, reconstructs the underlying signal, and visualises the results as both static 2D plots and an interactive 3D spectral surface.

## Features

- FFT-based cycle detection and dominant frequency extraction
- STFT time-frequency analysis on log returns
- Noise filtering and signal reconstruction via inverse FFT
- Interactive 3D spectral surface visualization (Plotly)

## Visual Output

### 3D Mode -- Time-Frequency Surface

An interactive Plotly surface showing how spectral power evolves over time. Peaks represent dominant periodic behaviour at specific time windows.

Output: `output/spectral_3d_surface.html` (interactive) and `output/3d_preview.png` (static)

### 2D Mode -- Classical FFT Decomposition

Three-panel matplotlib plot:

1. **Price vs Filtered Signal** -- original price overlaid with reconstructed dominant cycles
2. **Frequency Spectrum** -- amplitude per frequency component
3. **Residual Noise** -- high-frequency component removed by the filter

Output: `output/spectral_decomposition.png`

## Tech Stack

| Component | Library |
|-----------|---------|
| Data | ccxt (Binance) |
| FFT / STFT | numpy, scipy |
| Data processing | pandas |
| 2D plots | matplotlib |
| 3D surface | plotly, kaleido |

## Getting Started

```bash
git clone https://github.com/f20250217-blip/crypto-spectral-engine.git
cd crypto-spectral-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Set `MODE = "2d"` or `MODE = "3d"` in `main.py` to switch visualization modes.

## Project Structure

```
crypto-spectral-engine/
├── src/
│   ├── data_fetcher.py          # Binance OHLCV retrieval
│   ├── spectral_analysis.py     # FFT + dominant frequency extraction
│   ├── signal_filter.py         # Noise filtering + inverse FFT
│   ├── visualizer.py            # 2D matplotlib plots
│   └── advanced_visualizer.py   # STFT + 3D Plotly surface
├── output/                      # Generated plots
├── main.py                      # Pipeline entry point
├── requirements.txt
└── README.md
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `TIMEFRAME` | `1h` | Candle interval |
| `LIMIT` | `500` | Number of candles |
| `TOP_N` | `5` | Dominant FFT components (2D mode) |
| `MODE` | `3d` | Visualization mode |

## License

MIT
