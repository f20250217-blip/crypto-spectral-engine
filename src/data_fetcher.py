"""
Data fetcher module: retrieves crypto OHLCV data from Binance via ccxt.
"""

import ccxt
import pandas as pd


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch OHLCV candle data from Binance.

    Args:
        symbol: Trading pair (default BTC/USDT).
        timeframe: Candle interval (default 1h).
        limit: Number of candles to retrieve (max 1000).

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except ccxt.BaseError as e:
        raise RuntimeError(f"Failed to fetch data from Binance: {e}") from e

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df
