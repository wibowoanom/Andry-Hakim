"""
screening/volatility_filter.py
Modul 3: Volatility Filter untuk Sistem Screening Scalping BEI (IDX).
(Versi Bersih Tanpa Circular Import)
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from screening.config import AppConfig, VolatilityConfig
from screening.data_fetcher import load_ohlcv

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Menghitung Average True Range (ATR) Wilder."""
    if len(df) < period:
        return pd.Series(np.nan, index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()

def percentile_rank(series: pd.Series, window: int = 60) -> pd.Series:
    """Menghitung ranking persentil bergulir."""
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def analyze_volatility(ticker: str, cfg: VolatilityConfig) -> dict:
    df = load_ohlcv(ticker, "1D", price_mode="raw")
    if df is None or len(df) < cfg.atr_period + 10:
        return {"ticker": ticker, "status_volatility": "FAIL", "atr_pct": 0.0, "atr_percentile": 0.0}
    
    if cfg.exclude_incomplete_last_bar and len(df) > 1:
        df = df.iloc[:-1]
        
    df["atr"] = atr(df, cfg.atr_period)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100.0
    df["atr_pct_rank"] = percentile_rank(df["atr_pct"], cfg.atr_percentile_lookback)
    
    last_atr_pct = float(df["atr_pct"].iloc[-1])
    last_rank = float(df["atr_pct_rank"].iloc[-1]) if pd.notna(df["atr_pct_rank"].iloc[-1]) else 0.5
    
    passed = (cfg.min_atr_pct <= last_atr_pct <= cfg.max_atr_pct)
    
    return {
        "ticker": ticker,
        "status_volatility": "PASS" if passed else "FAIL",
        "atr_pct": round(last_atr_pct, 2),
        "atr_percentile": round(last_rank, 2)
    }

def run_volatility_filter(tickers: Optional[List[str]] = None, config: Optional[AppConfig] = None) -> pd.DataFrame:
    cfg_app = config or AppConfig()
    cfg_vol = cfg_app.volatility
    target_tickers = tickers or cfg_app.default_universe
    
    results = [analyze_volatility(t.strip().upper().replace(".JK", ""), cfg_vol) for t in target_tickers]
    return pd.DataFrame(results)
