#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Factor-Augmented Inflation Forecasting Framework
--------------------------------------------------------
A production-style Streamlit application for inflation nowcasting,
forecasting, regime analysis, scenario simulation, diagnostics,
and policy support.

Corrected version:
- fixes regime probability length mismatch
- adds safer handling for sparse classes and empty scenario decompositions
- completes all tabs and reporting sections
"""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


st.set_page_config(
    page_title="Enhanced Inflation Forecasting Framework",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --rbz-navy: #0B1F3A;
    --rbz-blue: #174EA6;
    --rbz-gold: #D4A017;
    --rbz-bg: #F6F9FC;
    --rbz-card: rgba(255,255,255,0.96);
    --rbz-text: #132238;
    --rbz-muted: #6B7A90;
    --rbz-border: rgba(23,78,166,0.12);
    --rbz-shadow: 0 14px 34px rgba(11,31,58,0.10);
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(23,78,166,0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(212,160,23,0.11), transparent 18%),
        linear-gradient(180deg, #F7FAFD 0%, #EEF4FA 52%, #F8FBFE 100%);
    color: var(--rbz-text);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.6rem;
    max-width: 1550px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--rbz-navy) 0%, #102B4C 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #F5F8FC !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] > div {
    background: #FFFFFF !important;
    color: #132238 !important;
    border-radius: 12px !important;
}
.rbz-hero {
    background: linear-gradient(135deg, #0B1F3A 0%, #174EA6 58%, #D4A017 145%);
    padding: 28px 30px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 20px 44px rgba(11,31,58,0.22);
    margin-bottom: 18px;
}
.rbz-hero h1 {
    margin: 0 0 6px 0;
    font-size: 2.05rem;
    line-height: 1.1;
    font-weight: 800;
}
.rbz-hero p {
    margin: 0;
    font-size: 1rem;
    color: rgba(255,255,255,0.92);
}
.rbz-subline {
    margin-top: 10px;
    font-size: 0.92rem;
    color: rgba(255,255,255,0.88);
}
.metric-card {
    background: var(--rbz-card);
    border: 1px solid var(--rbz-border);
    border-radius: 22px;
    padding: 16px 18px;
    box-shadow: var(--rbz-shadow);
    min-height: 122px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 6px;
    background: linear-gradient(90deg, var(--rbz-blue), var(--rbz-gold));
}
.metric-title {
    font-size: 13px;
    color: var(--rbz-muted);
    margin-bottom: 8px;
    font-weight: 600;
    text-transform: uppercase;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--rbz-navy);
    line-height: 1.08;
}
.metric-note {
    font-size: 12px;
    color: var(--rbz-muted);
    margin-top: 8px;
}
.policy-card {
    background: linear-gradient(135deg, rgba(11,31,58,0.98) 0%, rgba(23,78,166,0.96) 100%);
    color: white;
    border-radius: 22px;
    padding: 20px 22px;
    box-shadow: 0 18px 38px rgba(11,31,58,0.18);
    margin: 10px 0 18px 0;
}
.small-card {
    background: rgba(255,255,255,0.95);
    border-radius: 18px;
    padding: 14px 16px;
    border: 1px solid rgba(23,78,166,0.08);
    box-shadow: 0 10px 24px rgba(11,31,58,0.06);
}
.footer-note {
    background: rgba(255,255,255,0.94);
    border-radius: 18px;
    padding: 16px 18px;
    border-left: 6px solid #D4A017;
    box-shadow: 0 10px 24px rgba(11,31,58,0.06);
}
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_DATA_PATH = "/Users/USER/Documents/Models and Programs/simulated_zimbabwe_inflation_data_augmented.csv"


SADC_LOWER = 3.0
SADC_UPPER = 7.0
FORECAST_HORIZONS = [1, 3, 6, 12]

TARGET_OPTIONS = {
    "Monthly Inflation": "monthly_inflation_from_cpi",
    "Annual Inflation": "annual_inflation_from_cpi",
    "Quarterly Inflation (QoQ)": "quarterly_inflation_qoq",
    "Quarterly Inflation (Annualised)": "quarterly_inflation_annualised",
    "Core Monthly Inflation": "core_monthly_inflation",
    "Core Annual Inflation": "core_annual_inflation",
    "Core Quarterly Inflation (QoQ)": "core_quarterly_inflation_qoq",
    "Core Quarterly Inflation (Annualised)": "core_quarterly_inflation_annualised",
}

BASE_DRIVER_COLS = [
    "exrate_change",
    "fuel_change",
    "m3_growth",
    "reserve_money_growth",
    "rtgs_growth",
    "mobile_growth",
    "pos_growth",
    "ppi_change",
    "cempi_change",
    "pdl_change",
    "policy_rate",
    "credit_private_sector_growth",
    "wage_growth_proxy",
    "output_gap_proxy",
    "fiscal_impulse_proxy",
    "inflation_expectations_proxy",
    "imported_inflation_proxy",
    "exchange_rate_volatility_3m",
    "inflation_diffusion_proxy",
]

FACTOR_BLOCKS = {
    "domestic_liquidity_factor": [
        "m3_growth",
        "reserve_money_growth",
        "rtgs_growth",
        "mobile_growth",
        "pos_growth",
        "credit_private_sector_growth",
        "policy_rate",
    ],
    "external_cost_factor": [
        "exrate_change",
        "exchange_rate_volatility_3m",
        "fuel_change",
        "imported_inflation_proxy",
        "cempi_change",
        "ppi_change",
    ],
    "inflation_momentum_factor": [
        "core_monthly_inflation",
        "core_annual_inflation",
        "inflation_expectations_proxy",
        "inflation_diffusion_proxy",
        "wage_growth_proxy",
    ],
    "demand_fiscal_factor": [
        "output_gap_proxy",
        "fiscal_impulse_proxy",
        "pdl_change",
    ],
}
FACTOR_NAMES = list(FACTOR_BLOCKS.keys())

REGIME_LABELS = [
    "Below Lower Band",
    "Within SADC Band",
    "Moderately Above Band",
    "Far Above Band",
]
REGIME_MAP = {label: i for i, label in enumerate(REGIME_LABELS)}
REVERSE_REGIME_MAP = {i: label for label, i in REGIME_MAP.items()}
ALL_REGIME_CODES = list(REVERSE_REGIME_MAP.keys())

SCENARIO_LIBRARY = {
    "Neutral": {
        "policy_rate": 0.0,
        "m3_growth": 0.0,
        "reserve_money_growth": 0.0,
        "exrate_change": 0.0,
        "fuel_change": 0.0,
        "fiscal_impulse_proxy": 0.0,
        "inflation_expectations_proxy": 0.0,
    },
    "Hawkish": {
        "policy_rate": 3.0,
        "m3_growth": -2.0,
        "reserve_money_growth": -2.0,
        "exrate_change": -1.0,
        "fuel_change": 0.0,
        "fiscal_impulse_proxy": -1.0,
        "inflation_expectations_proxy": -1.5,
    },
    "Dovish": {
        "policy_rate": -2.0,
        "m3_growth": 2.5,
        "reserve_money_growth": 2.0,
        "exrate_change": 1.0,
        "fuel_change": 0.0,
        "fiscal_impulse_proxy": 1.0,
        "inflation_expectations_proxy": 1.5,
    },
    "Exchange-Rate Shock": {
        "policy_rate": 0.0,
        "m3_growth": 0.0,
        "reserve_money_growth": 0.0,
        "exrate_change": 4.0,
        "fuel_change": 1.0,
        "fiscal_impulse_proxy": 0.0,
        "inflation_expectations_proxy": 2.0,
    },
    "Disinflation Support": {
        "policy_rate": 1.0,
        "m3_growth": -1.5,
        "reserve_money_growth": -1.0,
        "exrate_change": -0.8,
        "fuel_change": -0.5,
        "fiscal_impulse_proxy": -0.5,
        "inflation_expectations_proxy": -1.2,
    },
}


@dataclass
class HorizonResult:
    horizon: int
    metrics: pd.DataFrame
    backtest: pd.DataFrame
    prediction_table: pd.DataFrame
    feature_importance: pd.DataFrame
    contribution_table: pd.DataFrame
    revision_table: pd.DataFrame
    fitted_models: Dict[str, object]
    residual_std_map: Dict[str, float]
    best_model_name: str
    ensemble_pred: float
    regime_specific_pred: float
    horizon_forecast: float
    horizon_lower68: float
    horizon_upper68: float
    horizon_lower95: float
    horizon_upper95: float


@dataclass
class RegimeBundle:
    class_results: pd.DataFrame
    confusion: pd.DataFrame
    report: pd.DataFrame
    accuracy: float
    probabilities_latest: pd.DataFrame


def metric_card(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_date_column(date_series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(date_series, format="%d/%m/%y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(date_series.loc[missing], errors="coerce", dayfirst=True)
    return parsed


def sanitize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        if out[col].isna().all():
            out[col] = 0.0
        else:
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
            out[col] = out[col].fillna(out[col].median())
            out[col] = out[col].fillna(0.0)
    return out


@st.cache_data(show_spinner=False)
def safe_read_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    df.columns = df.columns.str.strip()
    if "date" not in df.columns:
        raise ValueError("The dataset must contain a 'date' column.")
    if "cpi" not in df.columns:
        raise ValueError("The dataset must contain a 'cpi' column.")
    df["date"] = parse_date_column(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    numeric_cols = [c for c in df.columns if c != "date"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def derive_cpi_measures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "monthly_inflation_from_cpi" not in out.columns:
        out["monthly_inflation_from_cpi"] = (out["cpi"] / out["cpi"].shift(1) - 1.0) * 100.0
    if "annual_inflation_from_cpi" not in out.columns:
        out["annual_inflation_from_cpi"] = (out["cpi"] / out["cpi"].shift(12) - 1.0) * 100.0
    if "quarterly_inflation_qoq" not in out.columns:
        out["quarterly_inflation_qoq"] = (out["cpi"] / out["cpi"].shift(3) - 1.0) * 100.0
    if "quarterly_inflation_annualised" not in out.columns:
        out["quarterly_inflation_annualised"] = ((out["cpi"] / out["cpi"].shift(3)) ** 4 - 1.0) * 100.0
    if "monthly_inflation_3mma" not in out.columns:
        out["monthly_inflation_3mma"] = out["monthly_inflation_from_cpi"].rolling(3).mean()
    if "monthly_inflation_6mma" not in out.columns:
        out["monthly_inflation_6mma"] = out["monthly_inflation_from_cpi"].rolling(6).mean()
    return out


def ensure_core_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "core_cpi" not in out.columns:
        nonfood = out["nonfood_cpi"] if "nonfood_cpi" in out.columns else out["cpi"] * 0.98
        nontrad = out["nontradables_cpi"] if "nontradables_cpi" in out.columns else out["cpi"] * 0.99
        trad = out["tradables_cpi"] if "tradables_cpi" in out.columns else out["cpi"] * 1.00
        admin = out["administered_price_cpi"] if "administered_price_cpi" in out.columns else out["cpi"] * 1.01
        out["core_cpi"] = 0.35 * nonfood + 0.30 * nontrad + 0.20 * trad + 0.15 * admin
    if "core_monthly_inflation" not in out.columns:
        out["core_monthly_inflation"] = (out["core_cpi"] / out["core_cpi"].shift(1) - 1.0) * 100.0
    if "core_annual_inflation" not in out.columns:
        out["core_annual_inflation"] = (out["core_cpi"] / out["core_cpi"].shift(12) - 1.0) * 100.0
    if "core_quarterly_inflation_qoq" not in out.columns:
        out["core_quarterly_inflation_qoq"] = (out["core_cpi"] / out["core_cpi"].shift(3) - 1.0) * 100.0
    if "core_quarterly_inflation_annualised" not in out.columns:
        out["core_quarterly_inflation_annualised"] = ((out["core_cpi"] / out["core_cpi"].shift(3)) ** 4 - 1.0) * 100.0
    return out


def add_missing_driver_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "policy_rate" not in out.columns:
        yoy = out["annual_inflation_from_cpi"].bfill().fillna(0)
        mom = out["monthly_inflation_from_cpi"].fillna(0)
        out["policy_rate"] = np.maximum(5, 12 + 0.06 * yoy + 0.03 * mom)
    if "credit_private_sector_growth" not in out.columns:
        m3g = out["m3_growth"] if "m3_growth" in out.columns else pd.Series(0, index=out.index)
        reserve_g = out["reserve_money_growth"] if "reserve_money_growth" in out.columns else pd.Series(0, index=out.index)
        out["credit_private_sector_growth"] = 0.55 * m3g.fillna(0) + 0.15 * reserve_g.fillna(0)
    if "wage_growth_proxy" not in out.columns:
        out["wage_growth_proxy"] = 0.60 * out["monthly_inflation_from_cpi"].fillna(0) + 0.20 * out["core_monthly_inflation"].fillna(0)
    if "output_gap_proxy" not in out.columns:
        m3g = out["m3_growth"] if "m3_growth" in out.columns else pd.Series(0, index=out.index)
        out["output_gap_proxy"] = m3g.fillna(0).rolling(3).mean().fillna(0) / 10.0
    if "fiscal_impulse_proxy" not in out.columns:
        reserve_g = out["reserve_money_growth"] if "reserve_money_growth" in out.columns else pd.Series(0, index=out.index)
        m3g = out["m3_growth"] if "m3_growth" in out.columns else pd.Series(0, index=out.index)
        out["fiscal_impulse_proxy"] = 0.35 * reserve_g.fillna(0) + 0.30 * m3g.fillna(0)
    if "exchange_rate_volatility_3m" not in out.columns:
        out["exchange_rate_volatility_3m"] = out["exrate_change"].rolling(3).std() if "exrate_change" in out.columns else 0.0
    if "inflation_expectations_proxy" not in out.columns:
        exr = out["exrate_change"] if "exrate_change" in out.columns else pd.Series(0, index=out.index)
        fuel = out["fuel_change"] if "fuel_change" in out.columns else pd.Series(0, index=out.index)
        m3g = out["m3_growth"] if "m3_growth" in out.columns else pd.Series(0, index=out.index)
        out["inflation_expectations_proxy"] = (
            0.30 * exr.fillna(0)
            + 0.20 * fuel.fillna(0)
            + 0.15 * m3g.fillna(0)
            + 0.20 * out["monthly_inflation_from_cpi"].shift(1).fillna(0)
            + 0.15 * out["core_monthly_inflation"].fillna(0)
        )
    if "imported_inflation_proxy" not in out.columns:
        exr = out["exrate_change"] if "exrate_change" in out.columns else pd.Series(0, index=out.index)
        fuel = out["fuel_change"] if "fuel_change" in out.columns else pd.Series(0, index=out.index)
        ppi = out["ppi_change"] if "ppi_change" in out.columns else pd.Series(0, index=out.index)
        cempi = out["cempi_change"] if "cempi_change" in out.columns else pd.Series(0, index=out.index)
        out["imported_inflation_proxy"] = 0.45 * exr.fillna(0) + 0.25 * fuel.fillna(0) + 0.15 * ppi.fillna(0) + 0.15 * cempi.fillna(0)
    if "inflation_diffusion_proxy" not in out.columns:
        pieces = []
        for col in ["ppi_change", "core_monthly_inflation", "exrate_change", "fuel_change"]:
            if col in out.columns:
                s = out[col].fillna(0)
                denom = s.max() - s.min()
                normed = (s - s.min()) / denom if denom != 0 else s * 0
                pieces.append(normed)
        out["inflation_diffusion_proxy"] = 0.0 if len(pieces) == 0 else pd.concat(pieces, axis=1).mean(axis=1) * 100
    return out


def add_subindex_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "food_inflation_proxy" not in out.columns:
        fuel = out["fuel_change"] if "fuel_change" in out.columns else 0
        imported = out["imported_inflation_proxy"] if "imported_inflation_proxy" in out.columns else 0
        out["food_inflation_proxy"] = (
            0.45 * out["monthly_inflation_from_cpi"].fillna(0)
            + 0.25 * pd.Series(fuel, index=out.index).fillna(0)
            + 0.30 * pd.Series(imported, index=out.index).fillna(0)
        )
    if "nonfood_inflation_proxy" not in out.columns:
        out["nonfood_inflation_proxy"] = 0.65 * out["core_monthly_inflation"].fillna(0) + 0.35 * out["monthly_inflation_from_cpi"].fillna(0)
    if "tradables_inflation_proxy" not in out.columns:
        exr = out["exrate_change"] if "exrate_change" in out.columns else 0
        imported = out["imported_inflation_proxy"] if "imported_inflation_proxy" in out.columns else 0
        out["tradables_inflation_proxy"] = 0.5 * pd.Series(exr, index=out.index).fillna(0) + 0.5 * pd.Series(imported, index=out.index).fillna(0)
    if "nontradables_inflation_proxy" not in out.columns:
        out["nontradables_inflation_proxy"] = 0.5 * out["core_monthly_inflation"].fillna(0) + 0.5 * out["wage_growth_proxy"].fillna(0)
    return out


def add_required_lags(df: pd.DataFrame, cols: List[str], max_lag: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            for lag in range(1, max_lag + 1):
                lag_col = f"{col}_lag{lag}"
                if lag_col not in out.columns:
                    out[lag_col] = out[col].shift(lag)
    return out


def classify_policy_regime(yoy: float) -> str:
    if pd.isna(yoy):
        return "Below Lower Band"
    if yoy < SADC_LOWER:
        return "Below Lower Band"
    if yoy <= SADC_UPPER:
        return "Within SADC Band"
    if yoy <= 15:
        return "Moderately Above Band"
    return "Far Above Band"


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = derive_cpi_measures(out)
    out = ensure_core_series(out)
    out = add_missing_driver_proxies(out)
    out = add_subindex_proxies(out)
    lag_targets = list(TARGET_OPTIONS.values()) + BASE_DRIVER_COLS + FACTOR_NAMES + [
        "food_inflation_proxy",
        "nonfood_inflation_proxy",
        "tradables_inflation_proxy",
        "nontradables_inflation_proxy",
    ]
    out = add_required_lags(out, lag_targets, max_lag=3)
    out["policy_regime"] = out["annual_inflation_from_cpi"].apply(classify_policy_regime)
    out["policy_regime_code"] = out["policy_regime"].map(REGIME_MAP)
    return out


def generate_data_quality_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    numeric_cols = [c for c in df.columns if c != "date"]
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        missing_pct = float(s.isna().mean() * 100)
        zero_pct = float((s == 0).mean() * 100) if len(s) else np.nan
        std = float(s.std()) if s.notna().sum() > 1 else np.nan
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).mean() * 100 if pd.notna(iqr) else 0
        stale_run = 0
        current = 1
        vals = s.ffill().tolist()
        for i in range(1, len(vals)):
            current = current + 1 if vals[i] == vals[i - 1] else 1
            stale_run = max(stale_run, current)
        rows.append(
            {
                "variable": col,
                "missing_pct": round(missing_pct, 2),
                "zero_pct": round(zero_pct, 2),
                "std_dev": round(std, 4) if pd.notna(std) else np.nan,
                "outlier_pct_iqr": round(float(outliers), 2),
                "longest_constant_run": stale_run,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["missing_pct", "outlier_pct_iqr"], ascending=False)
    headline = pd.DataFrame(
        {
            "metric": [
                "Observations",
                "Variables",
                "Start date",
                "End date",
                "Average missing % across variables",
                "Variables with >20% missing",
            ],
            "value": [
                len(df),
                len(numeric_cols),
                str(pd.to_datetime(df["date"]).min().date()) if "date" in df.columns else "N/A",
                str(pd.to_datetime(df["date"]).max().date()) if "date" in df.columns else "N/A",
                round(summary["missing_pct"].mean(), 2) if not summary.empty else 0,
                int((summary["missing_pct"] > 20).sum()) if not summary.empty else 0,
            ],
        }
    )
    return headline, summary


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    return s.clip(lower=s.quantile(lower_q), upper=s.quantile(upper_q))


def winsorize_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = winsorize_series(out[c])
    return out


def varimax(phi: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-6):
    p, k = phi.shape
    r = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        lam = np.dot(phi, r)
        u, s, vh = np.linalg.svd(np.dot(phi.T, lam**3 - (gamma / p) * np.dot(lam, np.diag(np.diag(lam.T @ lam)))))
        r = u @ vh
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return phi @ r, r


def extract_single_block_factor(df: pd.DataFrame, cols: List[str], factor_name: str, smooth_window: int = 3):
    usable_cols = [c for c in cols if c in df.columns]
    if len(usable_cols) < 2:
        empty = pd.Series(np.nan, index=df.index, name=factor_name)
        return empty, pd.DataFrame(), pd.DataFrame(), {}
    work = sanitize_numeric_frame(winsorize_df(df[usable_cols].copy(), usable_cols))
    valid_idx = work.dropna(how="all").index
    if len(valid_idx) < 12:
        empty = pd.Series(np.nan, index=df.index, name=factor_name)
        return empty, pd.DataFrame(), pd.DataFrame(), {}
    work_valid = work.loc[valid_idx]
    scaler = StandardScaler()
    X = scaler.fit_transform(work_valid)
    n_components = min(len(usable_cols), 3)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T
    rotated_loadings, rotation_matrix = varimax(loadings)
    rotated_scores = scores @ rotation_matrix
    factor = pd.Series(np.nan, index=df.index, name=factor_name)
    factor.loc[valid_idx] = rotated_scores[:, 0]
    factor = factor.rolling(smooth_window, min_periods=1).mean()
    loadings_df = pd.DataFrame(
        rotated_loadings,
        index=usable_cols,
        columns=[f"PC{i+1}" for i in range(rotated_loadings.shape[1])],
    ).reset_index().rename(columns={"index": "variable"})
    loadings_df.insert(0, "factor_block", factor_name)
    explained_df = pd.DataFrame(
        {
            "factor_block": factor_name,
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    meta = {
        "usable_cols": usable_cols,
        "scaler": scaler,
        "pca": pca,
        "rotation_matrix": rotation_matrix,
        "smooth_window": smooth_window,
        "history": factor.dropna().tolist(),
    }
    return factor, loadings_df, explained_df, meta


def build_central_bank_factor_block(df: pd.DataFrame):
    out = df.copy()
    loadings_list, explained_list, factor_meta = [], [], {}
    for factor_name, cols in FACTOR_BLOCKS.items():
        try:
            factor_series, loadings_df, explained_df, meta = extract_single_block_factor(out, cols, factor_name, smooth_window=3)
            out[factor_name] = factor_series
            if not loadings_df.empty:
                loadings_list.append(loadings_df)
            if not explained_df.empty:
                explained_list.append(explained_df)
            factor_meta[factor_name] = meta
        except Exception:
            out[factor_name] = np.nan
            factor_meta[factor_name] = {}
    loadings_all = pd.concat(loadings_list, ignore_index=True) if loadings_list else pd.DataFrame()
    explained_all = pd.concat(explained_list, ignore_index=True) if explained_list else pd.DataFrame()
    out = add_required_lags(out, FACTOR_NAMES, max_lag=3)
    return out, loadings_all, explained_all, factor_meta


def compute_factor_from_single_row(row: pd.Series, factor_name: str, factor_meta: Dict) -> float:
    meta = factor_meta.get(factor_name, {})
    usable_cols = meta.get("usable_cols", [])
    scaler = meta.get("scaler")
    pca = meta.get("pca")
    rotation_matrix = meta.get("rotation_matrix")
    smooth_window = meta.get("smooth_window", 3)
    history = meta.get("history", [])
    if not usable_cols or scaler is None or pca is None or rotation_matrix is None:
        return 0.0
    vals = [pd.to_numeric(row.get(c, 0.0), errors="coerce") for c in usable_cols]
    x = np.array(vals, dtype=float).reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    z = scaler.transform(x)
    raw_scores = pca.transform(z)
    rotated_scores = raw_scores @ rotation_matrix
    current_factor = float(rotated_scores[0, 0])
    if smooth_window <= 1:
        return current_factor
    hist = history[-(smooth_window - 1):] if len(history) >= (smooth_window - 1) else history
    return float(np.mean(hist + [current_factor])) if hist else current_factor


def build_feature_set(df: pd.DataFrame, target_col: str, horizon: int) -> Tuple[pd.DataFrame, List[str], str]:
    out = df.copy()
    if target_col not in out.columns:
        raise ValueError(f"Target column '{target_col}' is missing.")
    horizon_target = f"{target_col}_h{horizon}"
    out[horizon_target] = out[target_col].shift(-horizon)
    predictor_cols = [c for c in FACTOR_NAMES if c in out.columns]
    direct_controls = [
        "policy_rate",
        "exrate_change",
        "fuel_change",
        "fiscal_impulse_proxy",
        "inflation_expectations_proxy",
        "imported_inflation_proxy",
        "exchange_rate_volatility_3m",
        "food_inflation_proxy",
        "nonfood_inflation_proxy",
        "tradables_inflation_proxy",
        "nontradables_inflation_proxy",
    ]
    predictor_cols += [c for c in direct_controls if c in out.columns]
    for lag in range(1, 4):
        lag_col = f"{target_col}_lag{lag}"
        if lag_col in out.columns:
            predictor_cols.append(lag_col)
    for factor_col in FACTOR_NAMES:
        for lag in range(1, 4):
            lag_col = f"{factor_col}_lag{lag}"
            if lag_col in out.columns:
                predictor_cols.append(lag_col)
    predictor_cols = list(dict.fromkeys(predictor_cols))
    needed = ["date", "policy_regime", "policy_regime_code", target_col, horizon_target] + predictor_cols
    model_df = out[needed].copy()
    numeric_cols = [c for c in model_df.columns if c not in ["date", "policy_regime"]]
    model_df[numeric_cols] = sanitize_numeric_frame(model_df[numeric_cols])
    model_df = model_df.dropna(subset=["date", horizon_target]).reset_index(drop=True)
    if model_df.empty:
        raise ValueError(f"No usable rows remain for target '{target_col}' at horizon {horizon}.")
    return model_df, predictor_cols, horizon_target


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_regression(y_true, y_pred, model_name: str) -> Dict[str, float]:
    return {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }


def get_model_registry() -> Dict[str, object]:
    registry = {
        "Elastic Net": ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, random_state=123, max_iter=20000),
        "Random Forest": RandomForestRegressor(n_estimators=400, random_state=123, min_samples_leaf=2),
        "SVR": SVR(kernel="rbf", C=5.0, epsilon=0.05),
        "OLS": LinearRegression(),
    }
    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = XGBRegressor(
            n_estimators=250,
            objective="reg:squarederror",
            random_state=123,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
        )
    return registry


def fit_single_model(name: str, model, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame):
    if name in ["Elastic Net", "XGBoost", "OLS"]:
        model.fit(x_train.values, y_train)
        pred_train = model.predict(x_train.values)
        pred_test = model.predict(x_test.values)
        pred_last = model.predict(x_test.tail(1).values)[0]
    else:
        model.fit(x_train, y_train)
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        pred_last = model.predict(x_test.tail(1))[0]
    return model, pred_train, pred_test, float(pred_last)


def compute_linear_contributions(model_name: str, model, x_row: pd.DataFrame, predictor_cols: List[str]) -> pd.DataFrame:
    if model_name not in ["Elastic Net", "OLS"]:
        return pd.DataFrame({
            "Component": ["Model Type"],
            "Contribution": [np.nan],
            "Note": ["Contribution breakdown shown only for linear benchmark models"],
        })
    coef = pd.Series(model.coef_, index=predictor_cols)
    contributions = coef * x_row.iloc[0][predictor_cols]
    intercept = float(model.intercept_)
    table = pd.DataFrame(
        {
            "Component": list(contributions.index) + ["Intercept"],
            "Contribution": list(contributions.values) + [intercept],
        }
    ).sort_values("Contribution", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    table["Note"] = ""
    return table


def compute_feature_importance(best_model_name: str, best_model, predictor_cols: List[str]) -> pd.DataFrame:
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=predictor_cols)
    elif hasattr(best_model, "coef_"):
        imp = pd.Series(np.abs(best_model.coef_), index=predictor_cols)
    else:
        imp = pd.Series(np.nan, index=predictor_cols)
    return imp.sort_values(ascending=False).reset_index().rename(columns={"index": "feature", 0: "importance"})


def build_ensemble_prediction(prediction_table: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    work = prediction_table.dropna(subset=["RMSE", "Prediction"]).copy()
    if work.empty:
        return np.nan, work
    work["weight"] = 1 / work["RMSE"].replace(0, np.nan)
    work["weight"] = work["weight"] / work["weight"].sum()
    ensemble = float((work["Prediction"] * work["weight"]).sum())
    return ensemble, work[["Model", "RMSE", "weight", "Prediction"]]


def fit_regime_specific_model(model_df: pd.DataFrame, predictor_cols: List[str], horizon_target: str) -> Tuple[float, str]:
    latest_regime = model_df["policy_regime"].iloc[-1]
    subset = model_df[model_df["policy_regime"] == latest_regime].copy()
    if len(subset) < 18:
        subset = model_df.copy()
        label = "Fallback: Full Sample"
    else:
        label = latest_regime
    x = sanitize_numeric_frame(subset[predictor_cols])
    y = pd.to_numeric(subset[horizon_target], errors="coerce").fillna(0).values
    model = LinearRegression()
    model.fit(x.values, y)
    pred = float(model.predict(sanitize_numeric_frame(model_df[predictor_cols].tail(1)).values)[0])
    return pred, label


def rolling_uncertainty(prediction_errors: pd.Series) -> float:
    e = prediction_errors.dropna()
    if len(e) >= 12:
        return float(e.tail(12).std(ddof=1))
    if len(e) >= 4:
        return float(e.std(ddof=1))
    return 0.5


def revision_decomposition_from_two_vectors(base_table: pd.DataFrame, scenario_table: pd.DataFrame) -> pd.DataFrame:
    merge = base_table[["Component", "Contribution"]].rename(columns={"Contribution": "Baseline"}).merge(
        scenario_table[["Component", "Contribution"]].rename(columns={"Contribution": "Scenario"}),
        on="Component",
        how="outer",
    ).fillna(0.0)
    merge["Revision"] = merge["Scenario"] - merge["Baseline"]
    return merge.sort_values("Revision", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def fit_horizon_models(model_df: pd.DataFrame, predictor_cols: List[str], horizon_target: str, horizon: int, train_ratio: float) -> HorizonResult:
    if len(model_df) < 30:
        raise ValueError(f"Need at least 30 usable observations for horizon {horizon}.")
    train_size = max(24, int(len(model_df) * train_ratio))
    if train_size >= len(model_df):
        train_size = len(model_df) - 1
    train_df = model_df.iloc[:train_size].copy()
    test_df = model_df.iloc[train_size:].copy()
    x_train = sanitize_numeric_frame(train_df[predictor_cols])
    x_test = sanitize_numeric_frame(test_df[predictor_cols])
    y_train = pd.to_numeric(train_df[horizon_target], errors="coerce").fillna(0).values
    y_test = pd.to_numeric(test_df[horizon_target], errors="coerce").fillna(0).values

    fitted, resid_std_map, prediction_rows, backtest_frames = {}, {}, [], []
    preds = {}
    for name, model in get_model_registry().items():
        try:
            fitted_model, pred_train, pred_test, pred_last = fit_single_model(name, model, x_train, y_train, x_test)
            preds[name] = pred_test
            fitted[name] = fitted_model
            resid_std_map[name] = float(np.std(y_train - pred_train, ddof=1)) if len(y_train) > 2 else 0.5
            metrics = evaluate_regression(y_test, pred_test, name)
            prediction_rows.append({"Model": name, **metrics, "Prediction": pred_last})
            backtest_frames.append(
                pd.DataFrame(
                    {
                        "date": test_df["date"].values,
                        "Actual": y_test,
                        "Predicted": pred_test,
                        "Model": name,
                    }
                )
            )
        except Exception:
            continue

    prediction_table = pd.DataFrame(prediction_rows).sort_values("RMSE").reset_index(drop=True)
    if prediction_table.empty:
        raise ValueError(f"No model could be estimated for horizon {horizon}.")

    best_model_name = prediction_table.iloc[0]["Model"]
    best_model = fitted[best_model_name]
    feature_importance = compute_feature_importance(best_model_name, best_model, predictor_cols)
    contribution_table = compute_linear_contributions(best_model_name, best_model, x_test.tail(1), predictor_cols)
    ensemble_pred, _ = build_ensemble_prediction(prediction_table)
    regime_pred, regime_label = fit_regime_specific_model(model_df, predictor_cols, horizon_target)
    prediction_table["Regime note"] = ""
    prediction_table.loc[prediction_table["Model"] == best_model_name, "Regime note"] = f"Best model | regime model based on: {regime_label}"

    best_pred = float(prediction_table.iloc[0]["Prediction"])
    combined = float(np.nanmean([best_pred, ensemble_pred, regime_pred]))
    errors_best = pd.Series(y_test - preds[best_model_name])
    sigma = max(rolling_uncertainty(errors_best), resid_std_map.get(best_model_name, 0.5), 0.25)

    lower68 = combined - sigma
    upper68 = combined + sigma
    lower95 = combined - 1.96 * sigma
    upper95 = combined + 1.96 * sigma

    backtest = pd.concat(backtest_frames, ignore_index=True) if backtest_frames else pd.DataFrame()
    best_subset = backtest[backtest["Model"] == best_model_name].copy()
    revision_table = pd.DataFrame(
        {
            "Component": ["Best model", "Ensemble", "Regime-specific", "Combined forecast"],
            "Contribution": [best_pred, ensemble_pred, regime_pred, combined],
        }
    )

    return HorizonResult(
        horizon=horizon,
        metrics=prediction_table,
        backtest=best_subset,
        prediction_table=prediction_table,
        feature_importance=feature_importance,
        contribution_table=contribution_table,
        revision_table=revision_table,
        fitted_models=fitted,
        residual_std_map=resid_std_map,
        best_model_name=best_model_name,
        ensemble_pred=ensemble_pred,
        regime_specific_pred=regime_pred,
        horizon_forecast=combined,
        horizon_lower68=lower68,
        horizon_upper68=upper68,
        horizon_lower95=lower95,
        horizon_upper95=upper95,
    )


def regime_classification_workflow(df: pd.DataFrame, predictor_cols: List[str]) -> RegimeBundle:
    reg_df = df.dropna(subset=["policy_regime_code"]).copy()
    if len(reg_df) < 24:
        empty = pd.DataFrame()
        return RegimeBundle(empty, empty, empty, np.nan, empty)

    x = sanitize_numeric_frame(reg_df[predictor_cols])
    y = reg_df["policy_regime_code"].astype(int)
    test_size = max(6, math.ceil(len(reg_df) * 0.25))
    train_size = len(reg_df) - test_size
    if train_size <= 0:
        empty = pd.DataFrame()
        return RegimeBundle(empty, empty, empty, np.nan, empty)

    x_train = x.iloc[:train_size]
    x_test = x.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    clf = RandomForestClassifier(
        n_estimators=350,
        random_state=123,
        min_samples_leaf=2,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)

    actual_labels = y_test.map(REVERSE_REGIME_MAP).values
    predicted_labels = pd.Series(pred).map(REVERSE_REGIME_MAP).fillna("Unknown").values
    class_results = pd.DataFrame(
        {
            "date": reg_df.iloc[train_size:]["date"].values,
            "Actual": actual_labels,
            "Predicted": predicted_labels,
        }
    )

    conf_array = confusion_matrix(y_test, pred, labels=ALL_REGIME_CODES)
    conf = pd.DataFrame(conf_array, index=REGIME_LABELS, columns=REGIME_LABELS)
    report = class_results.groupby(["Actual", "Predicted"]).size().reset_index(name="count")

    # Corrected probability block: align with trained classes, then expand to full regime space.
    model_classes = list(clf.classes_)
    probs = clf.predict_proba(x.tail(1))[0]
    prob_dict = dict(zip(model_classes, probs))
    full_probs = [float(prob_dict.get(code, 0.0)) for code in ALL_REGIME_CODES]
    latest_prob = pd.DataFrame(
        {
            "Regime": [REVERSE_REGIME_MAP[code] for code in ALL_REGIME_CODES],
            "Probability": full_probs,
        }
    ).sort_values("Probability", ascending=False).reset_index(drop=True)

    return RegimeBundle(class_results, conf, report, acc, latest_prob)


def apply_scenario_to_latest_row(df_factor: pd.DataFrame, factor_meta: Dict, shocks: Dict[str, float]) -> pd.Series:
    row = df_factor.iloc[-1].copy()
    for k, v in shocks.items():
        if k in row.index:
            row[k] = pd.to_numeric(row[k], errors="coerce") + float(v)
        else:
            row[k] = float(v)
    for factor_name in FACTOR_NAMES:
        row[factor_name] = compute_factor_from_single_row(row, factor_name, factor_meta)
    for factor_name in FACTOR_NAMES:
        row[f"{factor_name}_lag1"] = row.get(factor_name, 0.0)
    return row


def scenario_driver_decomposition(shocks: Dict[str, float], df_factor: pd.DataFrame) -> pd.DataFrame:
    latest = df_factor.iloc[-1]
    rows = []
    for var, shock in shocks.items():
        baseline = float(pd.to_numeric(latest.get(var, 0.0), errors="coerce"))
        scenario = baseline + float(shock)
        rows.append({"Variable": var, "Baseline": baseline, "Shock": float(shock), "Scenario": scenario})
    return pd.DataFrame(rows).sort_values("Shock", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def factor_revision_table(df_factor: pd.DataFrame, factor_meta: Dict, shocks: Dict[str, float]) -> pd.DataFrame:
    baseline_row = df_factor.iloc[-1].copy()
    scenario_row = apply_scenario_to_latest_row(df_factor, factor_meta, shocks)
    rows = []
    for fac in FACTOR_NAMES:
        b = float(pd.to_numeric(baseline_row.get(fac, 0.0), errors="coerce"))
        s = float(pd.to_numeric(scenario_row.get(fac, 0.0), errors="coerce"))
        rows.append({"Factor": fac, "Baseline": b, "Scenario": s, "Revision": s - b})
    return pd.DataFrame(rows).sort_values("Revision", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def scenario_forecasts_for_all_horizons(
    df_factor: pd.DataFrame,
    factor_meta: Dict,
    shocks: Dict[str, float],
    horizon_results: Dict[int, HorizonResult],
    predictor_cols_by_horizon: Dict[int, List[str]],
):
    scenario_row = apply_scenario_to_latest_row(df_factor, factor_meta, shocks)
    results = []
    revision_detail = {}
    for h, hr in horizon_results.items():
        predictor_cols = predictor_cols_by_horizon[h]
        x_row = pd.DataFrame([{c: scenario_row.get(c, 0.0) for c in predictor_cols}])
        x_row = sanitize_numeric_frame(x_row)
        pred_table_rows = []
        for name, model in hr.fitted_models.items():
            try:
                if name in ["Elastic Net", "XGBoost", "OLS"]:
                    pred = float(model.predict(x_row.values)[0])
                else:
                    pred = float(model.predict(x_row)[0])
                model_rmse = float(hr.metrics.loc[hr.metrics["Model"] == name, "RMSE"].iloc[0])
                pred_table_rows.append({"Model": name, "Prediction": pred, "RMSE": model_rmse})
            except Exception:
                continue
        pred_table = pd.DataFrame(pred_table_rows).sort_values("RMSE") if pred_table_rows else pd.DataFrame()
        if pred_table.empty:
            continue
        best_model_name = pred_table.iloc[0]["Model"]
        best_model = hr.fitted_models[best_model_name]
        scenario_contrib = compute_linear_contributions(best_model_name, best_model, x_row, predictor_cols)
        revision_detail[h] = revision_decomposition_from_two_vectors(hr.contribution_table, scenario_contrib)
        weighted_ensemble, _ = build_ensemble_prediction(pred_table)
        scenario_best = float(pred_table.iloc[0]["Prediction"])
        scenario_combined = float(np.nanmean([scenario_best, weighted_ensemble]))
        sigma = max(hr.horizon_upper68 - hr.horizon_forecast, 0.25)
        results.append(
            {
                "Horizon": h,
                "Baseline": hr.horizon_forecast,
                "Scenario": scenario_combined,
                "Revision": scenario_combined - hr.horizon_forecast,
                "Lower68": scenario_combined - sigma,
                "Upper68": scenario_combined + sigma,
                "Lower95": scenario_combined - 1.96 * sigma,
                "Upper95": scenario_combined + 1.96 * sigma,
            }
        )
    return pd.DataFrame(results), scenario_row, revision_detail


def line_chart(df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_layout(template="plotly_white", height=420)
    return fig


def factor_path_chart(df_factor: pd.DataFrame):
    plot_df = df_factor[["date"] + [c for c in FACTOR_NAMES if c in df_factor.columns]].copy()
    plot_df = plot_df.melt(id_vars="date", var_name="Factor", value_name="Value")
    fig = px.line(plot_df, x="date", y="Value", color="Factor", title="Factor Paths")
    fig.update_layout(template="plotly_white", height=420)
    return fig


def forecast_horizon_chart(horizon_results: Dict[int, HorizonResult], title: str):
    df = pd.DataFrame(
        {
            "Horizon": list(horizon_results.keys()),
            "Forecast": [v.horizon_forecast for v in horizon_results.values()],
            "Lower68": [v.horizon_lower68 for v in horizon_results.values()],
            "Upper68": [v.horizon_upper68 for v in horizon_results.values()],
        }
    ).sort_values("Horizon")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Horizon"], y=df["Upper68"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df["Horizon"], y=df["Lower68"], mode="lines", line=dict(width=0), fill="tonexty", name="68% band"))
    fig.add_trace(go.Scatter(x=df["Horizon"], y=df["Forecast"], mode="lines+markers", name="Forecast"))
    fig.update_layout(template="plotly_white", title=title, xaxis_title="Months ahead", yaxis_title="Inflation")
    return fig


def backtest_chart(backtest_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["Actual"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["Predicted"], mode="lines", name="Predicted"))
    fig.update_layout(template="plotly_white", title=title, height=400)
    return fig


def regime_prob_chart(prob_df: pd.DataFrame):
    fig = px.bar(prob_df, x="Regime", y="Probability", title="Current Regime Probabilities")
    fig.update_layout(template="plotly_white", height=400)
    return fig


def scenario_comparison_chart(base_vs_scenario: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_vs_scenario["Horizon"], y=base_vs_scenario["Baseline"], mode="lines+markers", name="Baseline"))
    fig.add_trace(go.Scatter(x=base_vs_scenario["Horizon"], y=base_vs_scenario["Scenario"], mode="lines+markers", name="Scenario"))
    fig.update_layout(template="plotly_white", title=title, xaxis_title="Months ahead", yaxis_title="Inflation")
    return fig


def contribution_bar(table: pd.DataFrame, title: str, label_col: str = "Component"):
    t = table.copy().head(12)
    col = "Revision" if "Revision" in t.columns else "Contribution"
    fig = px.bar(t, x=col, y=label_col, orientation="h", title=title)
    fig.update_layout(template="plotly_white", height=420, yaxis={"categoryorder": "total ascending"})
    return fig


def data_quality_heat(summary: pd.DataFrame):
    work = summary.head(20).copy()
    fig = px.bar(work, x="variable", y="missing_pct", title="Top variables by missingness")
    fig.update_layout(template="plotly_white", height=380)
    return fig


def interpret_revision(x: float) -> str:
    if x > 0.75:
        return "material upward revision"
    if x > 0.15:
        return "mild upward revision"
    if x < -0.75:
        return "material downward revision"
    if x < -0.15:
        return "mild downward revision"
    return "broadly unchanged"


def scenario_summary_text(scenario_name: str, scenario_df: pd.DataFrame, latest_regime: str, target_label: str) -> str:
    if scenario_df.empty:
        return "Scenario results could not be generated."
    h1 = float(scenario_df.loc[scenario_df["Horizon"] == 1, "Revision"].iloc[0]) if (scenario_df["Horizon"] == 1).any() else 0.0
    h12 = float(scenario_df.loc[scenario_df["Horizon"] == 12, "Revision"].iloc[0]) if (scenario_df["Horizon"] == 12).any() else 0.0
    return (
        f"Under the {scenario_name} scenario, the near-term {target_label.lower()} path shows a {interpret_revision(h1)} at the 1-month horizon "
        f"and a {interpret_revision(h12)} at the 12-month horizon. Current conditions place inflation in the '{latest_regime}' regime. "
        "This scenario should be interpreted as a conditional policy stress test rather than a deterministic forecast."
    )


def policy_brief_text(target_label: str, latest_actual: float, horizon_results: Dict[int, HorizonResult], latest_regime: str) -> str:
    h1 = horizon_results[1].horizon_forecast if 1 in horizon_results else np.nan
    h12 = horizon_results[12].horizon_forecast if 12 in horizon_results else np.nan
    direction = "easing" if h12 < latest_actual else "rising" if h12 > latest_actual else "stable"
    return (
        f"Latest observed {target_label.lower()} stands at {latest_actual:,.2f}%. The model system indicates a near-term 1-month forecast of {h1:,.2f}% "
        f"and a 12-month forecast of {h12:,.2f}%, suggesting a {direction} inflation profile. Inflation is currently classified as '{latest_regime}'. "
        "The framework combines factor-augmented modelling, ensemble methods, and regime-aware benchmarking to support policy interpretation."
    )


def build_export_pack(
    df_factor: pd.DataFrame,
    metrics_pack: Dict[int, HorizonResult],
    scenario_df: pd.DataFrame,
    dq_summary: pd.DataFrame,
    factor_loadings: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_factor.to_excel(writer, sheet_name="prepared_data", index=False)
        for h, res in metrics_pack.items():
            res.metrics.to_excel(writer, sheet_name=f"h{h}_metrics", index=False)
            res.backtest.to_excel(writer, sheet_name=f"h{h}_backtest", index=False)
            res.feature_importance.to_excel(writer, sheet_name=f"h{h}_importance", index=False)
            res.contribution_table.to_excel(writer, sheet_name=f"h{h}_contrib", index=False)
        scenario_df.to_excel(writer, sheet_name="scenario_results", index=False)
        dq_summary.to_excel(writer, sheet_name="data_quality", index=False)
        factor_loadings.to_excel(writer, sheet_name="factor_loadings", index=False)
    output.seek(0)
    return output.getvalue()


st.markdown(
    """
    <div class="rbz-hero">
        <h1>Enhanced Factor-Augmented Inflation Forecasting Framework</h1>
        <p>Nowcasting, multi-horizon forecasting, regime analysis, scenario stress testing, diagnostics, and policy reporting.</p>
        <div class="rbz-subline">Central-bank style analytical workflow with factor blocks, ensembles, regime-sensitive views, and operational reporting.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload inflation dataset (CSV)", type=["csv"])
source_note = "Uploaded dataset" if uploaded_file is not None else f"Local file: {DEFAULT_DATA_PATH}"

selected_target_label = st.sidebar.selectbox("Forecast target", list(TARGET_OPTIONS.keys()), index=1)
selected_target_col = TARGET_OPTIONS[selected_target_label]
train_ratio = st.sidebar.slider("Training sample ratio", min_value=0.60, max_value=0.90, value=0.80, step=0.05)
scenario_name = st.sidebar.selectbox("Scenario preset", list(SCENARIO_LIBRARY.keys()), index=0)
run_custom = st.sidebar.checkbox("Override preset with custom scenario")

scenario_shocks = SCENARIO_LIBRARY[scenario_name].copy()
st.sidebar.markdown("### Scenario shocks")
for k in list(scenario_shocks.keys()):
    scenario_shocks[k] = st.sidebar.number_input(
        k,
        value=float(scenario_shocks[k]),
        step=0.25,
        format="%.2f",
        disabled=not run_custom,
    )

try:
    raw_df = safe_read_csv(uploaded_file if uploaded_file is not None else DEFAULT_DATA_PATH)
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

prepared_df = prepare_dataset(raw_df)
dq_headline, dq_summary = generate_data_quality_report(prepared_df)
df_factor, factor_loadings, factor_explained, factor_meta = build_central_bank_factor_block(prepared_df)
prepared_final = add_required_lags(df_factor, list(TARGET_OPTIONS.values()) + FACTOR_NAMES, max_lag=3)

horizon_results: Dict[int, HorizonResult] = {}
predictor_cols_by_horizon: Dict[int, List[str]] = {}
errors = []

for horizon in FORECAST_HORIZONS:
    try:
        model_df, predictor_cols, horizon_target = build_feature_set(prepared_final, selected_target_col, horizon)
        predictor_cols_by_horizon[horizon] = predictor_cols
        result = fit_horizon_models(model_df, predictor_cols, horizon_target, horizon, train_ratio)
        horizon_results[horizon] = result
    except Exception as exc:
        errors.append(f"Horizon {horizon}: {exc}")

if not horizon_results:
    st.error("The model could not estimate any forecast horizon. Please check data coverage and required variables.")
    st.stop()

regime_predictors = predictor_cols_by_horizon[min(predictor_cols_by_horizon.keys())]
regime_bundle = regime_classification_workflow(prepared_final.dropna(subset=[selected_target_col]), regime_predictors)
scenario_df, scenario_row, scenario_revision_detail = scenario_forecasts_for_all_horizons(
    prepared_final,
    factor_meta,
    scenario_shocks,
    horizon_results,
    predictor_cols_by_horizon,
)
factor_revision = factor_revision_table(prepared_final, factor_meta, scenario_shocks)
scenario_drivers = scenario_driver_decomposition(scenario_shocks, prepared_final)

latest_actual_target = float(pd.to_numeric(prepared_final[selected_target_col].iloc[-1], errors="coerce"))
latest_yoy = float(pd.to_numeric(prepared_final["annual_inflation_from_cpi"].iloc[-1], errors="coerce"))
latest_regime = classify_policy_regime(latest_yoy)
one_month = horizon_results.get(1)
current_best_model = one_month.best_model_name if one_month else list(horizon_results.values())[0].best_model_name
current_backtest_rmse = one_month.metrics.iloc[0]["RMSE"] if one_month else list(horizon_results.values())[0].metrics.iloc[0]["RMSE"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Latest Actual", f"{latest_actual_target:,.2f}%", selected_target_label)
with c2:
    metric_card("1-Month Forecast", f"{horizon_results[min(horizon_results.keys())].horizon_forecast:,.2f}%", current_best_model)
with c3:
    metric_card("Backtest RMSE", f"{current_backtest_rmse:,.2f}", "Lower is better")
with c4:
    metric_card("Current Regime", latest_regime, f"Annual inflation: {latest_yoy:.2f}%")

policy_text = policy_brief_text(selected_target_label, latest_actual_target, horizon_results, latest_regime)
scenario_text = scenario_summary_text(scenario_name, scenario_df, latest_regime, selected_target_label)

st.markdown(
    f"""
    <div class="policy-card">
        <h3>Monetary Policy Brief</h3>
        <p><b>Data source:</b> {source_note}</p>
        <p><b>Selected target:</b> {selected_target_label}</p>
        <p><b>Forecast workflow:</b> Multi-horizon, factor-augmented, ensemble-supported, regime-aware.</p>
        <p>{policy_text}</p>
        <p><b>Scenario note:</b> {scenario_text}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if errors:
    with st.expander("Pipeline warnings", expanded=False):
        for err in errors:
            st.warning(err)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Executive Dashboard",
        "Factor Engine",
        "Forecasting & Horizons",
        "Scenario Lab",
        "Policy Regimes",
        "Diagnostics",
        "Data Quality",
        "Downloads & Reports",
        "Documentation, Disclaimer & Ownership",
    ]
)

with tab1:
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(line_chart(prepared_final[["date", selected_target_col]].dropna(), "date", selected_target_col, f"Historical Path – {selected_target_label}"), use_container_width=True, key="historical_path_chart")
    with right:
        st.plotly_chart(forecast_horizon_chart(horizon_results, f"Multi-Horizon Forecast Term Structure – {selected_target_label}"), use_container_width=True, key="forecast_term_structure_chart")

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(factor_path_chart(prepared_final), use_container_width=True, key="factor_path_chart_main")
    with right:
        if not scenario_df.empty:
            st.plotly_chart(scenario_comparison_chart(scenario_df, "Baseline vs Scenario Forecast Path"), use_container_width=True, key="baseline_vs_scenario_chart")
        else:
            st.info("Scenario comparison not available.")

    summary_forecasts = pd.DataFrame(
        {
            "Horizon (months)": list(horizon_results.keys()),
            "Forecast": [v.horizon_forecast for v in horizon_results.values()],
            "Lower 68": [v.horizon_lower68 for v in horizon_results.values()],
            "Upper 68": [v.horizon_upper68 for v in horizon_results.values()],
            "Lower 95": [v.horizon_lower95 for v in horizon_results.values()],
            "Upper 95": [v.horizon_upper95 for v in horizon_results.values()],
            "Best model": [v.best_model_name for v in horizon_results.values()],
        }
    ).sort_values("Horizon (months)")
    st.markdown("### Forecast Summary")
    st.dataframe(summary_forecasts, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Factor Block Overview")
    st.plotly_chart(factor_path_chart(prepared_final), use_container_width=True, key="factor_path_chart_decomposition")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Factor Loadings")
        if not factor_loadings.empty:
            st.dataframe(factor_loadings, use_container_width=True, hide_index=True)
        else:
            st.info("Factor loadings could not be extracted.")
    with right:
        st.markdown("### Explained Variance")
        if not factor_explained.empty:
            st.dataframe(factor_explained, use_container_width=True, hide_index=True)
        else:
            st.info("Explained variance table is unavailable.")

    st.markdown("### Factor Revision Under Scenario")
    st.dataframe(factor_revision, use_container_width=True, hide_index=True)
    st.plotly_chart(contribution_bar(factor_revision.rename(columns={"Factor": "Component"}), "Scenario-Induced Factor Revisions"), use_container_width=True, key="scenario_factor_revisions_chart")

with tab3:
    horizon_choice = st.selectbox("Choose forecast horizon for detailed diagnostics", sorted(horizon_results.keys()), index=0)
    res = horizon_results[horizon_choice]

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(backtest_chart(res.backtest, f"Backtest – {horizon_choice}-Month Horizon | {res.best_model_name}"), use_container_width=True, key=f"backtest_chart_{horizon_choice}")
    with right:
        st.plotly_chart(contribution_bar(res.feature_importance.rename(columns={"feature": "Component", "importance": "Contribution"}), f"Feature Importance – {horizon_choice}-Month Horizon"), use_container_width=True, key=f"feature_importance_chart_{horizon_choice}")

    st.markdown("### Model Comparison Table")
    st.dataframe(res.metrics, use_container_width=True, hide_index=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Linear Contribution Breakdown")
        st.dataframe(res.contribution_table, use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Forecast Combination Breakdown")
        st.dataframe(res.revision_table, use_container_width=True, hide_index=True)

with tab4:
    st.subheader(f"Scenario Results – {scenario_name}")
    st.dataframe(scenario_drivers, use_container_width=True, hide_index=True)

    if not scenario_df.empty:
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(scenario_comparison_chart(scenario_df, f"Scenario Path – {scenario_name}"), use_container_width=True, key=f"scenario_path_chart_{scenario_name}")
        with right:
            st.plotly_chart(contribution_bar(scenario_df.rename(columns={"Horizon": "Component", "Revision": "Contribution"}), "Forecast Revision by Horizon"), use_container_width=True, key=f"forecast_revision_by_horizon_{scenario_name}")
        st.markdown("### Scenario Forecast Table")
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        if scenario_revision_detail:
            chosen_rev_h = st.selectbox("Choose horizon for scenario revision decomposition", sorted(scenario_revision_detail.keys()), index=0)
            st.markdown(f"### Revision Decomposition – Horizon {chosen_rev_h} months")
            st.dataframe(scenario_revision_detail[chosen_rev_h], use_container_width=True, hide_index=True)
            st.plotly_chart(contribution_bar(scenario_revision_detail[chosen_rev_h], f"Revision Drivers – Horizon {chosen_rev_h}"), use_container_width=True, key=f"revision_drivers_chart_{chosen_rev_h}")
        else:
            st.info("Detailed scenario revision decomposition is unavailable for the current selection.")
    else:
        st.info("Scenario results could not be generated.")

with tab5:
    st.subheader("Policy Regime Analysis")
    if not regime_bundle.probabilities_latest.empty:
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(regime_prob_chart(regime_bundle.probabilities_latest), use_container_width=True, key="regime_probability_chart")
        with right:
            st.markdown("### Latest Regime Probability Table")
            st.dataframe(regime_bundle.probabilities_latest, use_container_width=True, hide_index=True)
    else:
        st.info("Regime probabilities are unavailable.")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Classification Accuracy")
        if pd.notna(regime_bundle.accuracy):
            st.metric("Accuracy", f"{100 * regime_bundle.accuracy:,.2f}%")
        else:
            st.info("Insufficient data for regime classification.")
    with c2:
        st.markdown("### Confusion Matrix")
        if not regime_bundle.confusion.empty:
            st.dataframe(regime_bundle.confusion, use_container_width=True)
        else:
            st.info("Confusion matrix unavailable.")

    st.markdown("### Actual vs Predicted Regime States")
    if not regime_bundle.class_results.empty:
        st.dataframe(regime_bundle.class_results, use_container_width=True, hide_index=True)
    else:
        st.info("No regime classification backtest available.")

with tab6:
    st.subheader("Diagnostics")
    diag_horizon = st.selectbox("Choose horizon for diagnostics", sorted(horizon_results.keys()), key="diag_h")
    diag_res = horizon_results[diag_horizon]

    if not diag_res.backtest.empty:
        residuals = diag_res.backtest.copy()
        residuals["Residual"] = residuals["Actual"] - residuals["Predicted"]
        left, right = st.columns([1, 1])
        with left:
            fig = px.line(residuals, x="date", y="Residual", title=f"Residual Path – Horizon {diag_horizon}")
            fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap_chart")
        with right:
            fig = px.histogram(residuals, x="Residual", nbins=20, title=f"Residual Distribution – Horizon {diag_horizon}")
            fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig, use_container_width=True, key="scatter_matrix_chart")

        summary = pd.DataFrame(
            {
                "Metric": ["Mean residual", "Residual std. dev.", "Min residual", "Max residual"],
                "Value": [
                    residuals["Residual"].mean(),
                    residuals["Residual"].std(ddof=1),
                    residuals["Residual"].min(),
                    residuals["Residual"].max(),
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Backtest diagnostics unavailable.")

with tab7:
    st.subheader("Data Quality")
    left, right = st.columns([0.9, 1.1])
    with left:
        st.dataframe(dq_headline, use_container_width=True, hide_index=True)
    with right:
        st.plotly_chart(data_quality_heat(dq_summary), use_container_width=True, key="data_quality_heatmap")
    st.markdown("### Variable-Level Data Quality Table")
    st.dataframe(dq_summary, use_container_width=True, hide_index=True)

with tab8:
    st.subheader("Downloads and Reporting")
    export_bytes = build_export_pack(prepared_final, horizon_results, scenario_df, dq_summary, factor_loadings)
    st.download_button(
        label="Download analytical Excel pack",
        data=export_bytes,
        file_name="inflation_framework_analytical_pack.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    report_text = f"""
Enhanced Factor-Augmented Inflation Forecasting Framework

Selected target: {selected_target_label}
Current actual: {latest_actual_target:,.2f}%
Current regime: {latest_regime}
One-month forecast: {horizon_results[min(horizon_results.keys())].horizon_forecast:,.2f}%
Twelve-month forecast: {horizon_results[12].horizon_forecast:,.2f}%
Scenario selected: {scenario_name}

Policy brief:
{policy_text}

Scenario note:
{scenario_text}
""".strip()

    st.download_button(
        label="Download policy note (.txt)",
        data=report_text.encode("utf-8"),
        file_name="inflation_policy_note.txt",
        mime="text/plain",
    )

with tab9:
    st.title("Documentation, Disclaimer and Ownership Notice")

    with st.expander("1. What this enhanced version adds", expanded=True):
        st.markdown(
            """
This version extends the baseline framework into a more complete policy-support system by adding:

- multi-horizon forecasting for **1, 3, 6, and 12 months**;
- model comparison across **Elastic Net, OLS, Random Forest, SVR, and XGBoost** where available;
- **ensemble forecast combination**;
- **regime-sensitive forecasting** and regime probability assessment;
- **scenario-driven forecast revision analysis**;
- **factor revision decomposition**;
- **data quality monitoring**;
- downloadable analytical packs and policy note exports.
            """
        )

    with st.expander("2. Multi-horizon forecasting logic", expanded=False):
        st.markdown(r"""
For each horizon $h \in \{1,3,6,12\}$, the application estimates:

$$
\pi_{t+h} = f_h(X_t, F_t, L\pi_t, LF_t) + \varepsilon_{t+h}
$$

where:
- $\pi_{t+h}$ is inflation at horizon $h$,
- $X_t$ is the vector of raw and proxy macro drivers,
- $F_t$ is the factor block vector,
- $L\pi_t$ are lags of inflation,
- $LF_t$ are lags of the factor blocks.
        """)

    with st.expander("3. Ensemble forecast combination", expanded=False):
        st.markdown(r"""
Candidate models are estimated separately and then combined using inverse-RMSE weights:

$$
w_m = \frac{1 / RMSE_m}{\sum_j (1 / RMSE_j)}
$$

$$
\hat{\pi}^{ens}_{t+h} = \sum_m w_m \hat{\pi}^{(m)}_{t+h}
$$

The operational forecast reported by the app averages the best single-model prediction,
the weighted ensemble forecast, and the regime-specific forecast.
        """)

    with st.expander("4. Scenario engine mathematics", expanded=False):
        st.markdown(r"""
The scenario engine follows the transmission path:

$$
s_t \rightarrow X_t^{(s)} \rightarrow F_t^{(s)} \rightarrow \widetilde{Z}_t^{(s)} \rightarrow \hat{\pi}_{t+h}^{(s)}
$$

where:
- $s_t$ is the vector of user-imposed shocks,
- $X_t^{(s)}$ is the shocked raw driver vector,
- $F_t^{(s)}$ is the recomputed factor vector,
- $\widetilde{Z}_t^{(s)}$ is the final scenario feature vector,
- $\hat{\pi}_{t+h}^{(s)}$ is the scenario-conditioned forecast.
        """)

    with st.expander("5. Regime classification rules", expanded=False):
        st.markdown(r"""
Annual inflation is classified as:

$$
\text{Regime}_t =
\begin{cases}
\text{Below Lower Band}, & \pi_t^y < 3 \\
\text{Within SADC Band}, & 3 \leq \pi_t^y \leq 7 \\
\text{Moderately Above Band}, & 7 < \pi_t^y \leq 15 \\
\text{Far Above Band}, & \pi_t^y > 15
\end{cases}
$$
        """)

    with st.expander("6. Disclaimer", expanded=False):
        st.markdown(
            """
This application has been developed as a technical analytical and policy-support tool for inflation nowcasting,
forecasting, scenario simulation, and regime assessment.

The outputs generated by this system are model-based estimates derived from historical data, statistical procedures,
latent factor extraction, and machine-learning forecasting techniques. Results should therefore be interpreted as
**conditional analytical outputs** rather than official forecasts, binding policy positions, or guaranteed future outcomes.

The framework remains subject to limitations relating to data quality, proxy construction, structural breaks,
and model uncertainty.
            """
        )

    with st.expander("7. Ownership and intellectual attribution", expanded=False):
        st.markdown(
            """
**Author / Intellectual Owner**

**Chirume A.T.**  
**Ph.D\*, MA, MSc, BSc (Hons) Econ (UZ)**

**Further Enquiries**  
Mobile: **+263 773 369 884**
            """
        )

st.markdown(
    """
        """,
    unsafe_allow_html=True,
)
