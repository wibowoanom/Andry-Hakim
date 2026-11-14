"""
screening/liquidity_filter.py
Modul 2: Liquidity Filter untuk Sistem Screening Scalping BEI (IDX).
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from screening.config import AppConfig, LiquidityConfig, get_bei_tick_size
from screening.data_fetcher import load_ohlcv

def analyze_liquidity(ticker: str, cfg: LiquidityConfig) -> dict:
    df = load_ohlcv(ticker, "1D", price_mode="raw")
    if df is None or len(df) < cfg.lookback_days:
        return {"ticker": ticker, "status_liquidity": "FAIL", "adv_idr": 0, "avg_vol_lot": 0, "spread_pct": 99.0}
    
    sub = df.tail(cfg.lookback_days)
    close_prices = sub["close"]
    volumes = sub["volume"]
    
    # Hitung ADV (Average Daily Value dalam Rupiah)
    daily_value = close_prices * volumes * 100 # 1 lot = 100 lembar saham di BEI
    adv_idr = float(daily_value.mean())
    avg_vol_lot = float(volumes.mean())
    
    # Estimasi spread berdasarkan tick size terakhir
    last_price = float(close_prices.iloc[-1])
    tick_size = get_bei_tick_size(last_price)
    spread_pct = float((tick_size / last_price) * 100.0)
    
    passed = (adv_idr >= cfg.min_adv_value_idr) and (avg_vol_lot >= cfg.min_avg_volume_lot) and (spread_pct <= cfg.max_spread_pct)
    
    return {
        "ticker": ticker,
        "status_liquidity": "PASS" if passed else "FAIL",
        "adv_idr": adv_idr,
        "avg_vol_lot": avg_vol_lot,
        "spread_pct": round(spread_pct, 2)
    }

def run_liquidity_filter(tickers: Optional[List[str]] = None, config: Optional[AppConfig] = None) -> pd.DataFrame:
    cfg_app = config or AppConfig()
    cfg_liq = cfg_app.liquidity
    target_tickers = tickers or cfg_app.default_universe
    
    results = [analyze_liquidity(t.strip().upper().replace(".JK", ""), cfg_liq) for t in target_tickers]
    df_res = pd.DataFrame(results)
    return df_res

def get_liquid_universe(result_df: pd.DataFrame) -> List[str]:
    if result_df.empty or "status_liquidity" not in result_df.columns:
        return []
    return result_df.loc[result_df["status_liquidity"] == "PASS", "ticker"].tolist()
