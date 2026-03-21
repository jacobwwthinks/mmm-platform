"""
Two-Step Calibrated OLS aMER Model.

Fits a spend → aMER model directly on observed monthly data, following the
DTC Forecast Tool specification (Sections 9–10).

    Step 1 — Cross-year spend elasticity:
        For each calendar month in 2+ years, compare aMER across years
        holding seasonality constant. β_log_spend = average elasticity.

    Step 2 — OLS on residual with fixed elasticity:
        aMER_adj = aMER - β_log_spend × LN(spend)
        OLS: aMER_adj ~ Intercept + β_trend × trend + β_promo × promo
                        + month_2 ... month_12  (January baseline)

    Prediction:
        aMER = β_log_spend × LN(spend) + Intercept + β_trend × trend
               + β_promo × promo + month_k

This model uses the same model_df as the MMM — no separate data file needed.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AMER_COEFF_FILENAME = "amer_coefficients.json"


# ═══════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def aggregate_weekly_to_monthly(
    model_df: pd.DataFrame,
    min_monthly_spend: float = 1000,
) -> pd.DataFrame:
    """
    Aggregate weekly model_df to monthly for aMER model training.

    Returns DataFrame with: year_month, year, month, trend_index,
    total_spend, total_revenue, aMER, promo, n_weeks.
    """
    mdf = model_df.copy()
    mdf["week_start"] = pd.to_datetime(mdf["week_start"])
    mdf["year_month"] = mdf["week_start"].dt.to_period("M")

    spend_cols = [c for c in mdf.columns if c.endswith("_spend")]
    mdf["total_spend"] = mdf[spend_cols].sum(axis=1)

    rev_col = "new_revenue" if "new_revenue" in mdf.columns else "revenue"

    # Promo flag: 1 if any week in the month has discount_campaign > 0
    has_promo = "discount_campaign" in mdf.columns

    monthly = mdf.groupby("year_month").agg(
        total_spend=("total_spend", "sum"),
        total_revenue=(rev_col, "sum"),
        n_weeks=("total_spend", "count"),
        promo_max=("discount_campaign", "max") if has_promo else ("total_spend", "count"),
    ).reset_index()

    if not has_promo:
        monthly["promo_max"] = 0

    monthly["promo"] = (monthly["promo_max"] > 0).astype(int)
    monthly["year"] = monthly["year_month"].apply(lambda p: p.year)
    monthly["month"] = monthly["year_month"].apply(lambda p: p.month)

    # Filter out months with negligible spend or partial months
    monthly = monthly[monthly["total_spend"] >= min_monthly_spend].copy()
    monthly = monthly[monthly["n_weeks"] >= 3].copy()

    monthly["aMER"] = monthly["total_revenue"] / monthly["total_spend"]
    monthly["log_spend"] = np.log(monthly["total_spend"])

    # Sequential trend index (0-based)
    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    monthly["trend_index"] = range(len(monthly))

    monthly["month_label"] = monthly["year_month"].astype(str)

    return monthly


# ═══════════════════════════════════════════════════════════════
# STEP 1 — CROSS-YEAR SPEND ELASTICITY
# ═══════════════════════════════════════════════════════════════

def compute_spend_elasticity(
    monthly_df: pd.DataFrame,
    min_log_spend_change: float = 0.20,
    default_elasticity: float = -0.15,
) -> dict:
    """
    Cross-year same-event spend elasticity (Step 1).

    For each calendar month appearing in 2+ years, compute:
        elasticity = LN(aMER_later / aMER_earlier) / LN(spend_later / spend_earlier)

    Filter: only pairs where |LN(spend ratio)| > min_log_spend_change.
    Returns dict with beta_log_spend and diagnostics.
    """
    df = monthly_df.copy()
    years = sorted(df["year"].unique())

    if len(years) < 2:
        logger.warning("Less than 2 years of data — using default spend elasticity")
        return {
            "beta_log_spend": default_elasticity,
            "n_valid_pairs": 0,
            "n_total_pairs": 0,
            "pairs": [],
            "used_default": True,
        }

    pairs = []
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            y1, y2 = years[i], years[j]
            d1 = df[df["year"] == y1]
            d2 = df[df["year"] == y2]

            for _, row2 in d2.iterrows():
                m = row2["month"]
                row1 = d1[d1["month"] == m]
                if row1.empty:
                    continue
                row1 = row1.iloc[0]

                log_spend_ratio = np.log(row2["total_spend"] / row1["total_spend"])
                if abs(log_spend_ratio) < min_log_spend_change:
                    continue

                log_amer_ratio = np.log(row2["aMER"] / row1["aMER"])
                elasticity = log_amer_ratio / log_spend_ratio

                pairs.append({
                    "month": m,
                    "year_from": y1,
                    "year_to": y2,
                    "spend_from": row1["total_spend"],
                    "spend_to": row2["total_spend"],
                    "aMER_from": row1["aMER"],
                    "aMER_to": row2["aMER"],
                    "elasticity": elasticity,
                })

    if not pairs:
        logger.warning("No valid elasticity pairs found — using default")
        return {
            "beta_log_spend": default_elasticity,
            "n_valid_pairs": 0,
            "n_total_pairs": 0,
            "pairs": [],
            "used_default": True,
        }

    elasticities = [p["elasticity"] for p in pairs]
    beta_log_spend = float(np.mean(elasticities))

    # Ensure negative (higher spend → lower aMER)
    if beta_log_spend > 0:
        logger.warning(
            f"Positive elasticity ({beta_log_spend:.4f}) — flipping sign. "
            "This may indicate unstable data; inspect the pairs."
        )
        beta_log_spend = -abs(beta_log_spend)

    return {
        "beta_log_spend": beta_log_spend,
        "n_valid_pairs": len(pairs),
        "n_total_pairs": len(pairs),
        "pairs": pairs,
        "used_default": False,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 2 — OLS ON RESIDUAL WITH FIXED ELASTICITY
# ═══════════════════════════════════════════════════════════════

def fit_amer_ols(
    monthly_df: pd.DataFrame,
    beta_log_spend: float,
) -> dict:
    """
    OLS on aMER residual after removing fixed spend elasticity (Step 2).

    aMER_adj = aMER - β_log_spend × LN(spend)
    OLS: aMER_adj ~ Intercept + β_trend × trend + β_promo × promo
                    + month_2 ... month_12

    Returns dict with all coefficients, R², and diagnostics.
    """
    df = monthly_df.copy()
    n = len(df)

    if n < 6:
        logger.warning(f"Only {n} months of data — aMER OLS may be unstable")

    # Dependent variable: aMER adjusted for spend elasticity
    df["aMER_adj"] = df["aMER"] - beta_log_spend * df["log_spend"]

    # Design matrix: intercept + trend + promo + 11 month dummies (Jan baseline)
    X_cols = ["intercept", "trend_index", "promo"]
    df["intercept"] = 1.0
    for m in range(2, 13):
        col_name = f"month_{m}"
        df[col_name] = (df["month"] == m).astype(float)
        X_cols.append(col_name)

    X = df[X_cols].values
    y = df["aMER_adj"].values

    # OLS
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    # Extract coefficients
    coeff_dict = {name: float(coeffs[i]) for i, name in enumerate(X_cols)}
    intercept = coeff_dict.pop("intercept")
    beta_trend = coeff_dict.pop("trend_index")
    beta_promo = coeff_dict.pop("promo")
    month_dummies = {
        str(m): coeff_dict[f"month_{m}"]
        for m in range(2, 13)
    }

    # R-squared on the full model (including fixed spend elasticity)
    y_pred_full = beta_log_spend * df["log_spend"].values + X @ coeffs
    ss_res = np.sum((df["aMER"].values - y_pred_full) ** 2)
    ss_tot = np.sum((df["aMER"].values - df["aMER"].mean()) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-8)

    # Adjusted R-squared
    p = len(X_cols) + 1  # +1 for beta_log_spend
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / max(n - p - 1, 1)

    return {
        "intercept": intercept,
        "beta_trend": beta_trend,
        "beta_promo": beta_promo,
        "month_dummies": month_dummies,
        "r_squared": float(r_squared),
        "adj_r_squared": float(adj_r_squared),
        "n_observations": n,
    }


# ═══════════════════════════════════════════════════════════════
# FULL MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

def fit_amer_model(
    model_df: pd.DataFrame,
    min_monthly_spend: float = 1000,
) -> dict:
    """
    Train the full two-step aMER model from weekly model_df.

    Returns a coefficient dict ready for prediction and JSON storage.
    """
    monthly = aggregate_weekly_to_monthly(model_df, min_monthly_spend)

    if len(monthly) < 3:
        raise ValueError(
            f"Only {len(monthly)} valid months — need at least 3 to train aMER model"
        )

    # Step 1: Cross-year spend elasticity
    elasticity_result = compute_spend_elasticity(monthly)
    beta_log_spend = elasticity_result["beta_log_spend"]

    # Step 2: OLS on residual
    ols_result = fit_amer_ols(monthly, beta_log_spend)

    # Training period metadata
    first_month = monthly["month_label"].iloc[0]
    last_month = monthly["month_label"].iloc[-1]
    trend_index_max = int(monthly["trend_index"].max())

    # Observed aMER range (for sanity checking predictions)
    min_amer_observed = float(monthly["aMER"].min())
    max_amer_observed = float(monthly["aMER"].max())
    mean_amer_observed = float(monthly["aMER"].mean())

    # Invariant promo months: months that are ALWAYS promo or NEVER promo
    # in training data. β_promo should not be applied to always-0 months.
    promo_by_month = monthly.groupby("month")["promo"].mean()
    invariant_always_0 = [int(m) for m in promo_by_month.index if promo_by_month[m] == 0]
    invariant_always_1 = [int(m) for m in promo_by_month.index if promo_by_month[m] == 1]

    coefficients = {
        # Step 1
        "beta_log_spend": beta_log_spend,
        "elasticity_n_pairs": elasticity_result["n_valid_pairs"],
        "elasticity_used_default": elasticity_result["used_default"],
        # Step 2
        "intercept": ols_result["intercept"],
        "beta_trend": ols_result["beta_trend"],
        "beta_promo": ols_result["beta_promo"],
        "month_dummies": ols_result["month_dummies"],
        # Fit quality
        "r_squared": ols_result["r_squared"],
        "adj_r_squared": ols_result["adj_r_squared"],
        "n_observations": ols_result["n_observations"],
        # Metadata
        "training_period": f"{first_month} to {last_month}",
        "trend_index_max": trend_index_max,
        "min_amer_observed": min_amer_observed,
        "max_amer_observed": max_amer_observed,
        "mean_amer_observed": mean_amer_observed,
        "invariant_promo_always_0": invariant_always_0,
        "invariant_promo_always_1": invariant_always_1,
    }

    logger.info(
        f"aMER model trained: β_log_spend={beta_log_spend:.4f}, "
        f"R²={ols_result['r_squared']:.3f}, "
        f"{ols_result['n_observations']} months, "
        f"elasticity pairs={elasticity_result['n_valid_pairs']}"
    )

    return coefficients


# ═══════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════

def predict_amer(
    coefficients: dict,
    spend: float,
    month: int,
    promo: int = 0,
    trend_index: int = None,
    min_floor: float = 0.3,
) -> float:
    """
    Predict aMER at a given spend level for a specific month.

    Uses the trained two-step model coefficients. Trend is frozen at the
    last training value by default (DTC Forecast Tool approach).

    Args:
        coefficients: Output of fit_amer_model()
        spend: Monthly total ad spend (SEK)
        month: Calendar month (1-12)
        promo: 0 or 1 — promo flag for this month
        trend_index: Sequential month index. Defaults to trend_index_max (frozen).
        min_floor: Minimum aMER prediction (prevents nonsensical values)
    """
    if spend <= 0:
        return 0.0

    if trend_index is None:
        trend_index = coefficients["trend_index_max"]

    # Don't apply beta_promo to months that were invariant (always non-promo)
    # in training data — the coefficient is not identified for those months
    effective_promo = promo
    if month in coefficients.get("invariant_promo_always_0", []):
        effective_promo = 0

    amer_raw = (
        coefficients["beta_log_spend"] * np.log(spend)
        + coefficients["intercept"]
        + coefficients["beta_trend"] * trend_index
        + coefficients["beta_promo"] * effective_promo
        + coefficients["month_dummies"].get(str(month), 0.0)
    )

    return max(min_floor, amer_raw)


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_amer_coefficients(coefficients: dict, results_dir: str) -> None:
    """Save aMER coefficients to JSON in the results directory."""
    path = Path(results_dir) / AMER_COEFF_FILENAME
    with open(path, "w") as f:
        json.dump(coefficients, f, indent=2)
    logger.info(f"aMER coefficients saved to {path}")


def load_amer_coefficients(results_dir: str) -> dict:
    """Load aMER coefficients from JSON. Returns None if not found."""
    path = Path(results_dir) / AMER_COEFF_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
