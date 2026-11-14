"""
screening/data_fetcher.py
Modul Pengunduhan & Pemuatan Data OHLCV untuk Pipeline AladdinFoundry.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import numpy as np
from screening.config import AppConfig

def load_ohlcv(ticker: str, timeframe: str = "1D", price_mode: str = "raw") -> Optional[pd.DataFrame]:
    """Memuat data OHLCV dari penyimpanan lokal Parquet di Google Drive atau buat dummy jika belum ada."""
    cfg = AppConfig()
    clean_ticker = ticker.upper().replace(".JK", "")
    
    file_path = cfg.raw_dir / timeframe / f"{clean_ticker}.parquet"
    
    if not file_path.exists():
        file_path_alt = cfg.raw_dir / f"{clean_ticker}.parquet"
        if file_path_alt.exists():
            file_path = file_path_alt
        else:
            return generate_dummy_ohlcv(clean_ticker)
            
    try:
        df = pd.read_parquet(file_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Gagal memuat data untuk {clean_ticker}: {e}")
        return generate_dummy_ohlcv(clean_ticker)

def generate_dummy_ohlcv(ticker: str) -> pd.DataFrame:
    """Menghasilkan data OHLCV sintetis yang valid untuk pengujian pipeline."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100, freq="B", tz="Asia/Jakarta")
    np.random.seed(abs(hash(ticker)) % (2**32))
    base_price = 5000.0 if ticker in ["BBCA", "BBRI"] else 1000.0
    returns = np.random.normal(0.0005, 0.015, len(dates))
    closes = base_price * np.cumprod(1 + returns)
    highs = closes * (1 + np.random.uniform(0.002, 0.01, len(dates)))
    lows = closes * (1 - np.random.uniform(0.002, 0.01, len(dates)))
    opens = lows + (highs - lows) * np.random.uniform(0.2, 0.8, len(dates))
    volumes = np.random.randint(50000, 500000, len(dates))
    
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    }, index=dates)
    return df
