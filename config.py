"""
screening/config.py
Pusat Konfigurasi dan Fungsi Pendukung Sistem Screening Scalping BEI (IDX).
"""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path
import pandas as pd

def get_bei_tick_size(price: float) -> int:
    """Menghitung besar tick size berdasarkan aturan Bursa Efek Indonesia (BEI)."""
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

@dataclass
class RangeSetupProfileParams:
    adx_max: float
    sr_dist_max_atr: float
    rsi_oversold: float
    rsi_overbought: float
    vwap_return_min_rate: float
    allow_literal_20_80: bool = False

@dataclass
class RangeSetupConfig:
    exclude_incomplete_last_bar: bool = True
    position_mode: str = "inside_range"
    allow_short: bool = False
    adx_period: int = 14
    adx_slope_bars: int = 4
    adx_rapid_rise_thresh: float = 2.0
    range_short_window: int = 10
    range_long_window: int = 20
    inside_range_tolerance_atr: float = 0.25
    lower_zone_max: float = 0.25
    upper_zone_min: float = 0.75
    pivot_left_bars: int = 3
    pivot_right_bars: int = 3
    sr_cluster_tolerance_atr: float = 0.4
    psych_level_weight: float = 0.5
    rsi_period: int = 14
    mr_weight_rsi: float = 0.35
    mr_weight_sr: float = 0.35
    mr_weight_candle: float = 0.30
    vwap_lookback_days: int = 5
    vwap_away_pct: float = 1.0
    vwap_return_bars: int = 6
    vp_lookback_days: int = 5
    vp_value_area_pct: float = 0.70
    vp_bin_atr_fraction: float = 0.1
    candle_lookback_bars: int = 3
    pinbar_wick_body_ratio: float = 2.0
    pinbar_opposite_wick_max: float = 0.3
    doji_body_range_max: float = 0.10
    volume_spike_period: int = 20
    volume_spike_multiplier: float = 1.8
    profiles: Dict[str, RangeSetupProfileParams] = field(default_factory=lambda: {
        "standard": RangeSetupProfileParams(adx_max=25.0, sr_dist_max_atr=1.5, rsi_oversold=30.0, rsi_overbought=70.0, vwap_return_min_rate=0.40),
        "strict": RangeSetupProfileParams(adx_max=20.0, sr_dist_max_atr=1.0, rsi_oversold=25.0, rsi_overbought=75.0, vwap_return_min_rate=0.50),
        "relaxed": RangeSetupProfileParams(adx_max=30.0, sr_dist_max_atr=2.0, rsi_oversold=35.0, rsi_overbought=65.0, vwap_return_min_rate=0.30)
    })

@dataclass
class LiquidityConfig:
    min_adv_value_idr: float = 1_000_000_000
    min_avg_volume_lot: int = 5000
    max_spread_pct: float = 1.5
    lookback_days: int = 20
    round_trip_cost_pct: float = 0.35

@dataclass
class VolatilityConfig:
    atr_period: int = 14
    hv_period: int = 20
    min_atr_pct: float = 1.0
    max_atr_pct: float = 8.0
    atr_percentile_lookback: int = 60
    exclude_incomplete_last_bar: bool = True

@dataclass
class RiskConfig:
    max_position_idr: float = 10_000_000
    max_drawdown_limit: float = 15.0
    allow_manual_override: bool = True
    manual_csv_path: str = "data/meta/risk_manual_override.csv"

@dataclass
class ScoringConfig:
    weight_liquidity: float = 0.20
    weight_volatility: float = 0.25
    weight_setup: float = 0.35
    weight_risk: float = 0.20

@dataclass
class PipelineConfig:
    default_profile: str = "standard"
    include_watch_in_candidates: bool = True
    save_outputs: bool = True

@dataclass
class AppConfig:
    root_dir: Path = Path("/content/drive/MyDrive/Scalping_IDX")
    data_dir: Path = Path("/content/drive/MyDrive/Scalping_IDX/data")
    meta_dir: Path = Path("/content/drive/MyDrive/Scalping_IDX/data/meta")
    raw_dir: Path = Path("/content/drive/MyDrive/Scalping_IDX/data/raw")
    output_dir: Path = Path("/content/drive/MyDrive/Scalping_IDX/data/output")
    default_universe: List[str] = field(default_factory=lambda: ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "ADRO", "ANTM"])
    range_setup: RangeSetupConfig = field(default_factory=RangeSetupConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
