"""
screening/scoring_ranking.py
Modul 6: Scoring & Ranking Engine untuk Sistem Screening Scalping BEI (IDX).
"""

import pandas as pd
import numpy as np
from typing import Optional
from screening.config import AppConfig, ScoringConfig

def calculate_scoring_and_ranking(df_merged: pd.DataFrame, config: Optional[AppConfig] = None) -> pd.DataFrame:
    cfg_app = config or AppConfig()
    cfg_score = cfg_app.scoring
    
    if df_merged.empty:
        return df_merged
        
    df = df_merged.copy()
    
    if "adv_idr" in df.columns:
        max_adv = df["adv_idr"].max()
        df["score_liquidity"] = (df["adv_idr"] / max_adv * 100.0).fillna(0) if max_adv > 0 else 50.0
    else:
        df["score_liquidity"] = 50.0
        
    if "atr_pct" in df.columns:
        df["score_volatility"] = (100.0 - (df["atr_pct"] - 4.0).abs() * 20.0).clip(0, 100).fillna(50.0)
    else:
        df["score_volatility"] = 50.0
        
    if "price_position_20" in df.columns:
        df["score_setup"] = ((1.0 - df["price_position_20"]) * 100.0).clip(0, 100).fillna(50.0)
    else:
        df["score_setup"] = 50.0
        
    if "status_risk" in df.columns:
        df["score_risk"] = df["status_risk"].apply(lambda x: 100.0 if x == "PASS" else 0.0)
    else:
        df["score_risk"] = 100.0
        
    df["composite_score"] = (
        df["score_liquidity"] * cfg_score.weight_liquidity +
        df["score_volatility"] * cfg_score.weight_volatility +
        df["score_setup"] * cfg_score.weight_setup +
        df["score_risk"] * cfg_score.weight_risk
    )
    
    if "status_final" in df.columns:
        df_passed = df[df["status_final"] == "PASS"].sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    else:
        df_passed = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
        
    return df_passed
