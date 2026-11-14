"""
screening/trade_planner.py
Modul Trade Planner & Risk Matrix: Menghitung Entry, SL, TP, dan Lot Sizing 
berdasarkan ATR 1D dan Support Struktur BEI.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from screening.config import AppConfig, get_bei_tick_size
from screening.data_fetcher import load_ohlcv
from screening.volatility_filter import atr

def generate_trade_plan(ticker: str, max_risk_idr: float = 500_000.0, risk_reward_ratio: float = 2.0) -> Dict[str, Any]:
    """
    Menghitung rencana trading lengkap untuk ticker tertentu dengan pembulatan entry 1 desimal.
    """
    clean_ticker = ticker.upper().replace(".JK", "")
    cfg = AppConfig()
    
    # 1. Muat data OHLCV 1D (Daily)
    df_1d = load_ohlcv(clean_ticker, "1D", price_mode="raw")
    if df_1d is None or len(df_1d) < 30:
        return {"ticker": clean_ticker, "error": "Data tidak mencukupi"}
        
    # Hitung ATR 1D (periode 14)
    df_1d["atr_14"] = atr(df_1d, period=14)
    
    last_row = df_1d.iloc[-1]
    current_price = float(last_row["close"])
    current_atr = float(last_row["atr_14"]) if not np.isnan(last_row["atr_14"]) else current_price * 0.02
    
    # 2. Tentukan Support Dinamis (20-bar lowest low)
    support_level = float(df_1d["low"].tail(20).min())
    
    # 3. Hitung Stop Loss (SL) berbasis ATR di bawah support
    sl_price_raw = support_level - (0.5 * current_atr)
    tick_size = get_bei_tick_size(current_price)
    sl_price = round(sl_price_raw / tick_size) * tick_size
    if sl_price >= current_price:
        sl_price = current_price - (2 * tick_size)
        
    # Bulatkan entry price menjadi 1 digit desimal
    entry_price = round(float(current_price), 1)
    
    risk_per_share = entry_price - sl_price
    if risk_per_share <= 0:
        risk_per_share = tick_size
        
    # 4. Hitung Take Profit (TP1 dan TP2)
    tp1_price = entry_price + (risk_per_share * 1.5)
    tp2_price = entry_price + (risk_per_share * risk_reward_ratio)
    
    tp1_price = round(tp1_price / tick_size) * tick_size
    tp2_price = round(tp2_price / tick_size) * tick_size
    
    # 5. Position Sizing
    shares_needed = max_risk_idr / risk_per_share
    lots_needed = int(np.floor(shares_needed / 100.0))
    if lots_needed < 1:
        lots_needed = 1
        
    total_capital_required = entry_price * lots_needed * 100
    
    return {
        "ticker": clean_ticker,
        "current_price": entry_price,
        "support_level": support_level,
        "atr_1d": round(current_atr, 2),
        "entry_price": entry_price,
        "stop_loss": sl_price,
        "risk_per_share": round(risk_per_share, 2),
        "risk_pct": round((risk_per_share / entry_price) * 100, 2),
        "tp1": tp1_price,
        "tp2": tp2_price,
        "recommended_lots": lots_needed,
        "total_capital_idr": total_capital_required
    }
