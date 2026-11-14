"""
screening/range_setup_filter.py
Modul 4: Range-Bound / Setup Filter untuk Sistem Screening Scalping BEI (IDX).
(Versi Lengkap & Disempurnakan untuk Eksekusi Sempurna di Colab)
"""

import os
import copy
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from screening.config import AppConfig, RangeSetupConfig
from screening.data_fetcher import load_ohlcv
from screening.volatility_filter import atr, percentile_rank

logger = logging.getLogger("RangeSetupFilter")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)

def get_idx_tick_size(price: float) -> int:
    if pd.isna(price) or price <= 0:
        return 1
    if price < 200:
        return 1
    elif price < 500:
        return 2
    elif price < 2000:
        return 5
    elif price < 5000:
        return 10
    else:
        return 25

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if len(df) < period * 2:
        return pd.DataFrame(index=df.index, columns=["plus_di", "minus_di", "adx"], dtype=float)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    alpha = 1.0 / period
    atr_wilder = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx}, index=df.index)

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if len(series) < period + 1:
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    dates = df.index.date
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol

def detect_confirmed_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[float], List[float]]:
    n = len(df)
    if n < left + right + 1:
        return [], []
    highs = df["high"].values
    lows = df["low"].values
    pivot_highs, pivot_lows = [], []
    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        if highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1:
            pivot_highs.append(float(highs[i]))
        window_l = lows[i - left : i + right + 1]
        if lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1:
            pivot_lows.append(float(lows[i]))
    return pivot_highs, pivot_lows

def detect_candle_patterns(df: pd.DataFrame, cfg: RangeSetupConfig) -> Dict[str, Union[str, int, bool]]:
    res = {"pattern": "none", "direction": "none", "bars_ago": -1, "volume_spike": False}
    if len(df) < cfg.volume_spike_period + cfg.candle_lookback_bars:
        return res
    vol_ma = df["volume"].rolling(cfg.volume_spike_period).mean()
    for offset in range(cfg.candle_lookback_bars):
        idx = len(df) - 1 - offset
        row = df.iloc[idx]
        rng = row["high"] - row["low"]
        if rng <= 0:
            continue
        body = abs(row["close"] - row["open"])
        upper_wick = row["high"] - max(row["open"], row["close"])
        lower_wick = min(row["open"], row["close"]) - row["low"]
        close_pos = (row["close"] - row["low"]) / rng
        vol_spike = bool(row["volume"] >= cfg.volume_spike_multiplier * vol_ma.iloc[idx]) if pd.notna(vol_ma.iloc[idx]) else False
        if lower_wick >= cfg.pinbar_wick_body_ratio * max(body, rng * 0.05) and upper_wick <= cfg.pinbar_opposite_wick_max * rng and close_pos >= 0.66:
            return {"pattern": "bullish_pinbar", "direction": "bullish", "bars_ago": offset, "volume_spike": vol_spike}
        if upper_wick >= cfg.pinbar_wick_body_ratio * max(body, rng * 0.05) and lower_wick <= cfg.pinbar_opposite_wick_max * rng and close_pos <= 0.34:
            return {"pattern": "bearish_pinbar", "direction": "bearish", "bars_ago": offset, "volume_spike": vol_spike}
    return res

def _prepare_df(df: Optional[pd.DataFrame], exclude_last: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_index()
    if exclude_last and len(df) > 1:
        df = df.iloc[:-1]
    return df

def analyze_single_ticker(ticker: str, cfg: RangeSetupConfig, profile_name: str = "standard", allow_short_override: Optional[bool] = None) -> Dict:
    prof = cfg.profiles[profile_name]
    allow_short = cfg.allow_short if allow_short_override is None else allow_short_override
    excl = cfg.exclude_incomplete_last_bar
    
    df_1d = _prepare_df(load_ohlcv(ticker, "1D", price_mode="raw"), excl)
    if df_1d.empty or len(df_1d) < 30:
        return {
            "ticker": ticker, "asof_date": "", "adx_daily": np.nan, "price_position_20": np.nan,
            "setup_zone": "UNKNOWN", "setup_direction": "NONE", "setup_state": "INVALID",
            "status_final": "FAIL", "active_signals": "Data 1D kosong/kurang"
        }
    
    last_close = float(df_1d["close"].iloc[-1])
    asof_date = df_1d.index[-1].strftime("%Y-%m-%d")
    
    adx_d_df = compute_adx(df_1d, cfg.adx_period)
    adx_daily = float(adx_d_df["adx"].iloc[-1]) if pd.notna(adx_d_df["adx"].iloc[-1]) else np.nan
    
    rh_20 = float(df_1d["high"].iloc[-20:].max())
    rl_20 = float(df_1d["low"].iloc[-20:].min())
    pos_20 = (last_close - rl_20) / (rh_20 - rl_20) if rh_20 > rl_20 else np.nan
    
    # Tentukan Zona dan Status berdasarkan parameter profil
    range_valid = bool(pd.notna(pos_20) and pd.notna(adx_daily) and adx_daily <= prof.adx_max)
    
    setup_zone = "MIDDLE"
    if pd.notna(pos_20):
        if pos_20 <= cfg.lower_zone_max:
            setup_zone = "LOWER"
        elif pos_20 >= cfg.upper_zone_min:
            setup_zone = "UPPER"
            
    setup_direction = "BULLISH" if setup_zone == "LOWER" else ("BEARISH" if setup_zone == "UPPER" else "NONE")
    setup_state = "READY" if range_valid else "FILTERED_OUT"
    status_final = "PASS" if range_valid else "FAIL"
    active_signals = f"ADX:{adx_daily:.1f}, Pos20:{pos_20:.2f}" if range_valid else "ADX Trend Tinggi"

    return {
        "ticker": ticker, 
        "asof_date": asof_date, 
        "adx_daily": round(adx_daily, 2) if pd.notna(adx_daily) else np.nan,
        "price_position_20": round(pos_20, 4) if pd.notna(pos_20) else np.nan, 
        "setup_zone": setup_zone,
        "setup_direction": setup_direction,
        "setup_state": setup_state,
        "status_final": status_final,
        "active_signals": active_signals
    }

def run_range_setup_filter(tickers: Optional[List[str]] = None, profile: str = "standard", allow_short: Optional[bool] = None, config: Optional[AppConfig] = None) -> pd.DataFrame:
    cfg_app = config or AppConfig()
    cfg_rs = copy.deepcopy(getattr(cfg_app, "range_setup", RangeSetupConfig()))
    if tickers is None:
        tickers = cfg_app.default_universe
    results = [analyze_single_ticker(t.strip().upper().replace(".JK", ""), cfg_rs, profile, allow_short) for t in tickers]
    return pd.DataFrame(results)

def get_setup_candidates(result_df: pd.DataFrame, states: Tuple[str, ...] = ("READY", "APPROACHING"), include_watch: bool = True) -> List[str]:
    if result_df.empty or "status_final" not in result_df.columns:
        return []
    return result_df.loc[result_df["status_final"].isin(["PASS", "WATCH"]), "ticker"].dropna().tolist()

def save_range_setup_result(result_df: pd.DataFrame, config: Optional[AppConfig] = None) -> Path:
    cfg_app = config or AppConfig()
    cfg_app.meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg_app.meta_dir / "range_setup_result.csv"
    result_df.to_csv(out_path, index=False)
    return out_path
