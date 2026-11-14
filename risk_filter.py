"""
screening/risk_filter.py
Modul 5: Risk Filter & Manual Override untuk Sistem Screening Scalping BEI (IDX).
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from screening.config import AppConfig, RiskConfig

def run_risk_filter(tickers: List[str], config: Optional[AppConfig] = None) -> pd.DataFrame:
    cfg_app = config or AppConfig()
    cfg_risk = cfg_app.risk
    
    manual_overrides = {}
    csv_path = cfg_app.root_dir / cfg_risk.manual_csv_path
    if csv_path.exists():
        try:
            df_man = pd.read_csv(csv_path)
            if "ticker" in df_man.columns and "status_risk" in df_man.columns:
                manual_overrides = dict(zip(df_man["ticker"], df_man["status_risk"]))
        except Exception:
            pass
            
    results = []
    for t in tickers:
        t_clean = t.strip().upper().replace(".JK", "")
        status = manual_overrides.get(t_clean, "PASS")
            
        results.append({
            "ticker": t_clean,
            "status_risk": status,
            "max_position_limit_idr": cfg_risk.max_position_idr
        })
        
    return pd.DataFrame(results)
