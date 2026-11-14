"""
screening/pipeline.py
Orkestrator Utama: Menjalankan Pipeline Screening Lengkap (Modul 1 s.d 6).
"""

import pandas as pd
from typing import List, Optional
from screening.config import AppConfig
from screening.liquidity_filter import run_liquidity_filter
from screening.volatility_filter import run_volatility_filter
from screening.range_setup_filter import run_range_setup_filter
from screening.risk_filter import run_risk_filter
from screening.scoring_ranking import calculate_scoring_and_ranking

def run_full_pipeline(tickers: Optional[List[str]] = None, profile: str = "standard", config: Optional[AppConfig] = None) -> pd.DataFrame:
    cfg_app = config or AppConfig()
    universe = tickers or cfg_app.default_universe
    
    print("🚀 Menjalankan Pipeline AladdinFoundry (Modul 1 - 6)...")
    
    # 1. Modul Likuiditas
    df_liq = run_liquidity_filter(universe, cfg_app)
    liquid_tickers = df_liq.loc[df_liq["status_liquidity"] == "PASS", "ticker"].tolist()
    
    if not liquid_tickers:
        print("⚠️ Tidak ada ticker yang lolos filter likuiditas.")
        return pd.DataFrame()
        
    # 2. Modul Volatilitas
    df_vol = run_volatility_filter(liquid_tickers, cfg_app)
    vol_tickers = df_vol.loc[df_vol["status_volatility"] == "PASS", "ticker"].tolist()
    
    if not vol_tickers:
        print("⚠️ Tidak ada ticker yang lolos filter volatilitas.")
        return pd.DataFrame()
        
    # 3 & 4. Modul Range & Setup Filter (Modul 4)
    df_setup = run_range_setup_filter(vol_tickers, profile=profile, config=cfg_app)
    
    # 5. Modul Risk Filter (Modul 5)
    df_risk = run_risk_filter(vol_tickers, cfg_app)
    
    # Gabungkan seluruh dataframe hasil filter
    df_merged = df_liq.merge(df_vol, on="ticker", how="left")
    df_merged = df_merged.merge(df_setup, on="ticker", how="left")
    df_merged = df_merged.merge(df_risk, on="ticker", how="left")
    
    # Tentukan status final gabungan
    df_merged["status_final"] = "PASS"
    for idx, row in df_merged.iterrows():
        if row.get("status_liquidity") != "PASS" or row.get("status_volatility") != "PASS" or row.get("status_final_setup") == "FAIL" or row.get("status_risk") != "PASS":
            df_merged.at[idx, "status_final"] = "FAIL"
            
    # 6. Scoring & Ranking (Modul 6)
    df_final = calculate_scoring_and_ranking(df_merged, cfg_app)
    
    # Simpan hasil akhir ke folder output
    cfg_app.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cfg_app.output_dir / "final_screening_results.csv"
    df_final.to_csv(out_csv, index=False)
    print(f"✅ Pipeline selesai! Hasil akhir disimpan di: {out_csv}")
    
    return df_final
