"""
Spend-aMER Model: Optimal spend planning for GP3 maximization.

Combines MMM saturation curves with unit economics (GM2%, 365D CLTV expansion)
to find the spend level that maximizes GP3 (gross profit after all variable
costs including marketing).

    GP3 = (New Customer Revenue × (1 + CLTV_expansion) × GM2%) − Marketing Spend

Two breakeven aMER thresholds:
    First-order breakeven aMER = 1 / GM2%
        (covers variable costs on the first transaction alone)
    365D breakeven aMER = 1 / ((1 + CLTV_expansion) × GM2%)
        (covers costs when you account for repeat purchases over 12 months)

The optimal spend is where marginal GP3 = 0: the point on the saturation
curve where one more SEK of spend produces exactly one SEK of 365D
contribution margin.
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize_scalar, minimize
from typing import Optional
import datetime

from model.mmm import geometric_adstock, MMMResults

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CHANNEL RESPONSE PREDICTION
# ═══════════════════════════════════════════════════════════════

def _get_adstock_training_mean(params: dict, n_weeks: int, roas_row) -> float:
    """
    Get the adstock training mean for correct saturation normalization.

    If stored during training (adstock_training_mean), use that.
    Otherwise estimate from average weekly spend and decay rate.
    """
    if "adstock_training_mean" in params:
        return params["adstock_training_mean"]

    # Fallback: estimate from average spend and decay
    avg_weekly_spend = roas_row["total_spend"] / max(n_weeks, 1)
    decay = params["adstock_decay"]
    return avg_weekly_spend / (1 - decay + 1e-8) * 0.85


def predict_channel_revenue(
    spend_weekly: float,
    params: dict,
    adstock_training_mean: float,
    n_sim_weeks: int = 13,
) -> float:
    """
    Predict average weekly channel revenue at a given constant spend level.

    Uses the stored training adstock mean for correct saturation normalization,
    avoiding the scale-invariance issue in hill_saturation().
    """
    if spend_weekly <= 0:
        return 0.0

    spend_series = np.full(n_sim_weeks, spend_weekly)
    adstocked = geometric_adstock(spend_series, params["adstock_decay"], max_lag=8)

    # Normalize by TRAINING mean (not the new series mean)
    x_norm = adstocked / (adstock_training_mean + 1e-8)

    # Hill saturation (manual — bypasses hill_saturation() which renormalizes)
    alpha = np.clip(params["saturation_alpha"], 0.01, 10.0)
    # Use end-of-training lam if available (accounts for saturation curve shift)
    # saturation_lam_end captures the current state of the saturation curve,
    # reflecting any expansion in channel capacity over time.
    lam = np.clip(params.get("saturation_lam_end", params["saturation_lam"]), 0.01, 10.0)
    exponent = np.clip(lam * np.power(x_norm, alpha), 0, 30)
    saturated = 1 - np.exp(-exponent)

    # Use most recent effectiveness (beta_end_raw) if available,
    # otherwise fall back to base beta_raw. beta_end_raw accounts for
    # time-varying channel effectiveness (e.g., creative quality improvements).
    beta = params.get("beta_end_raw", params["beta_raw"])

    # Average of last 4 weeks (after adstock has reached steady state)
    return beta * saturated[-4:].mean()


# ═══════════════════════════════════════════════════════════════
# MODEL CALIBRATION
# ═══════════════════════════════════════════════════════════════

def compute_calibration_factor(
    results: MMMResults,
    model_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compare model-predicted channel revenue at historical spend vs actual.

    The MMM saturation curves give the right SHAPE (diminishing returns)
    but may underestimate the overall LEVEL of channel effectiveness —
    especially when the model attributes too much to organic/seasonality.

    This function computes a calibration factor:
        calibration = actual_channel_contribution / predicted_channel_contribution

    If calibration > 1, the model underestimates channel value.
    If calibration < 1, the model overestimates.

    Returns dict with calibration factor and diagnostics.
    """
    params = results.channel_params
    roas_df = results.channel_roas
    channels = [ch for ch in params.keys() if ch != "email"]

    if not channels:
        return {"factor": 1.0, "predicted_amer": 0, "actual_amer": 0, "status": "no_channels"}

    # ── Predicted channel revenue at historical spend ──
    predicted_weekly_rev = 0
    total_weekly_spend = 0
    for ch in channels:
        row = roas_df[roas_df["channel"] == ch].iloc[0]
        avg_spend = row["total_spend"] / max(results.n_weeks, 1)
        adstock_mean = _get_adstock_training_mean(params[ch], results.n_weeks, row)
        rev = predict_channel_revenue(avg_spend, params[ch], adstock_mean)
        predicted_weekly_rev += rev
        total_weekly_spend += avg_spend

    # ── Actual channel revenue from model's own decomposition ──
    contrib_df = results.channel_contributions
    channel_cols = [c for c in contrib_df.columns if c not in ["week_start", "email"]]
    actual_weekly_contrib = contrib_df[channel_cols].sum(axis=1).mean()

    # ── Actual aMER from training data ──
    actual_amer = 0
    if model_df is not None:
        mdf = model_df.copy()
        rev_col = "new_revenue" if "new_revenue" in mdf.columns else "revenue"
        spend_cols = [c for c in mdf.columns if c.endswith("_spend")]
        total_rev = mdf[rev_col].sum()
        total_spend = mdf[spend_cols].sum().sum()
        actual_amer = total_rev / (total_spend + 1e-8)

    predicted_amer = (predicted_weekly_rev + results.baseline_contribution.mean()) / (total_weekly_spend + 1e-8)

    # Calibration factor: how much to scale predicted channel revenue
    if predicted_weekly_rev > 0:
        factor = actual_weekly_contrib / predicted_weekly_rev
    else:
        factor = 1.0

    # Clamp to reasonable range (0.5 - 2.5)
    factor = max(0.5, min(2.5, factor))

    logger.info(f"Calibration: predicted_weekly_rev={predicted_weekly_rev:.0f}, "
                f"actual_weekly_contrib={actual_weekly_contrib:.0f}, factor={factor:.2f}")

    return {
        "factor": float(factor),
        "predicted_weekly_channel_rev": float(predicted_weekly_rev),
        "actual_weekly_channel_contrib": float(actual_weekly_contrib),
        "predicted_amer": float(predicted_amer),
        "actual_amer": float(actual_amer),
        "total_weekly_spend": float(total_weekly_spend),
        "status": "ok",
    }


# ═══════════════════════════════════════════════════════════════
# DIMINISHING CALIBRATION
# ═══════════════════════════════════════════════════════════════

def _effective_calibration(
    calibration_factor: float,
    total_spend: float,
    historical_avg_spend: float,
) -> float:
    """
    Apply calibration with diminishing weight as spend moves beyond observed data.

    At historical average spend, full calibration is applied (we have evidence).
    Beyond 1.5x historical, calibration fades linearly toward 1.0.
    Beyond 3x historical, no calibration correction at all.

    This prevents the optimizer from extrapolating an aggressive calibration
    factor into spend ranges where we have no data.
    """
    if historical_avg_spend <= 0 or calibration_factor == 1.0:
        return calibration_factor

    spend_ratio = total_spend / (historical_avg_spend + 1e-8)

    # Full calibration up to 1.5x historical, then linear fade to 1.0 by 3x
    if spend_ratio <= 1.5:
        return calibration_factor
    elif spend_ratio >= 3.0:
        return 1.0
    else:
        # Linear interpolation: 1.5 → full, 3.0 → 1.0
        fade = (spend_ratio - 1.5) / 1.5  # 0 at 1.5x, 1 at 3.0x
        return calibration_factor + (1.0 - calibration_factor) * fade


# ═══════════════════════════════════════════════════════════════
# GP3 CURVE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_gp3_curve(
    results: MMMResults,
    gm2_pct: float,
    cltv_expansion_pct: float,
    organic_weekly_revenue: float = 0,
    seasonal_multiplier: float = 1.0,
    calibration_factor: float = 1.0,
    n_points: int = 150,
    max_spend_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Compute GP3 at various total spend levels.

    The calibration_factor scales predicted channel revenue to match
    actual historical performance. The saturation curves give the right
    shape (diminishing returns), and the calibration corrects the level.

    Returns DataFrame with spend, revenue, GP3 (first order + 365D), and aMER.
    """
    params = results.channel_params
    roas_df = results.channel_roas
    channels = [ch for ch in params.keys() if ch != "email"]

    current_spend = {}
    adstock_means = {}
    for ch in channels:
        row = roas_df[roas_df["channel"] == ch].iloc[0]
        current_spend[ch] = row["total_spend"] / results.n_weeks
        adstock_means[ch] = _get_adstock_training_mean(params[ch], results.n_weeks, row)

    current_total = sum(current_spend.values())
    alloc_ratios = {ch: s / (current_total + 1e-8) for ch, s in current_spend.items()}

    cltv_mult = 1 + cltv_expansion_pct / 100
    gm2_frac = gm2_pct / 100

    spend_levels = np.linspace(0, current_total * max_spend_mult, n_points)

    rows = []
    for total_spend in spend_levels:
        channel_rev = 0
        for ch in channels:
            ch_spend = total_spend * alloc_ratios.get(ch, 1 / len(channels))
            ch_rev = predict_channel_revenue(ch_spend, params[ch], adstock_means[ch])
            channel_rev += ch_rev

        # Apply calibration with diminishing weight beyond historical spend
        eff_cal = _effective_calibration(calibration_factor, total_spend, current_total)
        channel_rev *= eff_cal * seasonal_multiplier

        total_new_rev = organic_weekly_revenue + channel_rev
        rev_365d = total_new_rev * cltv_mult
        gp3_first_order = total_new_rev * gm2_frac - total_spend
        gp3_365d = rev_365d * gm2_frac - total_spend
        amer = total_new_rev / (total_spend + 1e-8) if total_spend > 0 else 0

        rows.append({
            "weekly_spend": total_spend,
            "monthly_spend": total_spend * 4.33,
            "paid_new_customer_revenue": channel_rev,
            "total_new_customer_revenue": total_new_rev,
            "revenue_365d": rev_365d,
            "gp3_first_order": gp3_first_order,
            "gp3_first_order_monthly": gp3_first_order * 4.33,
            "gp3_365d": gp3_365d,
            "gp3_365d_monthly": gp3_365d * 4.33,
            "amer": amer,
        })

    df = pd.DataFrame(rows)

    if len(df) > 1:
        d_rev = np.gradient(df["paid_new_customer_revenue"].values, df["weekly_spend"].values)
        df["marginal_roas"] = d_rev
    else:
        df["marginal_roas"] = 0

    return df


# ═══════════════════════════════════════════════════════════════
# OPTIMAL SPEND FINDING
# ═══════════════════════════════════════════════════════════════

def find_optimal_spend(
    results: MMMResults,
    gm2_pct: float,
    cltv_expansion_pct: float,
    organic_weekly_revenue: float = 0,
    seasonal_multiplier: float = 1.0,
    calibration_factor: float = 1.0,
    max_spend_mult: float = 3.0,
) -> dict:
    """
    Find the weekly spend level that maximizes 365D GP3.

    The calibration_factor is applied with diminishing weight beyond
    historical spend levels (see _effective_calibration).

    Returns dict with optimal spend, GP3 (both first-order and 365D),
    both breakeven aMER thresholds, channel allocation, and current comparison.
    """
    params = results.channel_params
    roas_df = results.channel_roas
    channels = [ch for ch in params.keys() if ch != "email"]

    current_spend = {}
    adstock_means = {}
    for ch in channels:
        row = roas_df[roas_df["channel"] == ch].iloc[0]
        current_spend[ch] = row["total_spend"] / results.n_weeks
        adstock_means[ch] = _get_adstock_training_mean(params[ch], results.n_weeks, row)

    current_total = sum(current_spend.values())
    alloc_ratios = {ch: s / (current_total + 1e-8) for ch, s in current_spend.items()}

    cltv_mult = 1 + cltv_expansion_pct / 100
    gm2_frac = gm2_pct / 100
    breakeven_amer_first_order = 1 / gm2_frac
    breakeven_amer_365d = 1 / (cltv_mult * gm2_frac)

    def _compute_at_spend(total_spend):
        """Compute revenue and GP3 at a given spend level."""
        channel_rev = 0
        for ch in channels:
            ch_spend = total_spend * alloc_ratios.get(ch, 1 / len(channels))
            ch_rev = predict_channel_revenue(ch_spend, params[ch], adstock_means[ch])
            channel_rev += ch_rev
        # Diminishing calibration: full at historical spend, fading beyond
        eff_cal = _effective_calibration(calibration_factor, total_spend, current_total)
        channel_rev *= eff_cal * seasonal_multiplier
        total_new_rev = organic_weekly_revenue + channel_rev
        gp3_first_order = total_new_rev * gm2_frac - total_spend
        gp3_365d = total_new_rev * cltv_mult * gm2_frac - total_spend
        return total_new_rev, channel_rev, gp3_first_order, gp3_365d

    def neg_gp3_365d(total_spend):
        if total_spend <= 0:
            return -(organic_weekly_revenue * cltv_mult * gm2_frac)
        _, _, _, gp3 = _compute_at_spend(total_spend)
        return -gp3

    result = minimize_scalar(
        neg_gp3_365d,
        bounds=(0, current_total * max_spend_mult),
        method="bounded",
    )

    optimal_spend = result.x
    at_upper_bound = optimal_spend >= current_total * max_spend_mult * 0.95
    opt_rev, opt_ch_rev, opt_gp3_fo, opt_gp3_365d = _compute_at_spend(optimal_spend)

    # Channel breakdown at optimal
    eff_cal_optimal = _effective_calibration(calibration_factor, optimal_spend, current_total)
    channel_allocation = {}
    for ch in channels:
        ch_spend = optimal_spend * alloc_ratios.get(ch, 1 / len(channels))
        ch_rev = predict_channel_revenue(ch_spend, params[ch], adstock_means[ch])
        ch_rev *= eff_cal_optimal * seasonal_multiplier
        channel_allocation[ch] = {
            "weekly_spend": ch_spend,
            "monthly_spend": ch_spend * 4.33,
            "weekly_revenue": ch_rev,
        }

    # Current comparison
    cur_rev, _, cur_gp3_fo, cur_gp3_365d = _compute_at_spend(current_total)

    return {
        "optimal_weekly_spend": optimal_spend,
        "optimal_monthly_spend": optimal_spend * 4.33,
        "optimal_gp3_first_order_weekly": opt_gp3_fo,
        "optimal_gp3_first_order_monthly": opt_gp3_fo * 4.33,
        "optimal_gp3_365d_weekly": opt_gp3_365d,
        "optimal_gp3_365d_monthly": opt_gp3_365d * 4.33,
        "new_customer_revenue_weekly": opt_rev,
        "new_customer_revenue_monthly": opt_rev * 4.33,
        "amer_at_optimal": opt_rev / (optimal_spend + 1e-8) if optimal_spend > 1 else 0,
        "breakeven_amer_first_order": breakeven_amer_first_order,
        "breakeven_amer_365d": breakeven_amer_365d,
        "channel_allocation": channel_allocation,
        "current_weekly_spend": current_total,
        "current_monthly_spend": current_total * 4.33,
        "current_gp3_first_order_weekly": cur_gp3_fo,
        "current_gp3_first_order_monthly": cur_gp3_fo * 4.33,
        "current_gp3_365d_weekly": cur_gp3_365d,
        "current_gp3_365d_monthly": cur_gp3_365d * 4.33,
        "current_amer": cur_rev / (current_total + 1e-8),
        "spend_change_pct": (optimal_spend - current_total) / (current_total + 1e-8) * 100,
        "at_upper_bound": at_upper_bound,
    }


# ═══════════════════════════════════════════════════════════════
# SEASONAL & EVENT INDICES (DATA-DRIVEN)
# ═══════════════════════════════════════════════════════════════

def compute_seasonal_indices(
    results: MMMResults,
    model_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute monthly seasonal efficiency indices from historical data.

    For each calendar month, computes the ratio of channel efficiency
    (contribution per SEK of spend) vs the overall average.

    Returns:
        Dict {month_number: efficiency_index} where 1.0 = average month
    """
    if model_df is None:
        return {m: 1.0 for m in range(1, 13)}

    contrib_df = results.channel_contributions.copy()
    contrib_df["week_start"] = pd.to_datetime(contrib_df["week_start"])
    contrib_df["month"] = contrib_df["week_start"].dt.month

    channel_cols = [c for c in contrib_df.columns
                    if c not in ["week_start", "month", "email"]]
    if not channel_cols:
        return {m: 1.0 for m in range(1, 13)}

    contrib_df["total_contrib"] = contrib_df[channel_cols].sum(axis=1)

    mdf = model_df.copy()
    mdf["week_start"] = pd.to_datetime(mdf["week_start"])
    mdf["month"] = mdf["week_start"].dt.month
    spend_cols = [c for c in mdf.columns if c.endswith("_spend")]
    mdf["total_spend"] = mdf[spend_cols].sum(axis=1)

    merged = contrib_df[["week_start", "month", "total_contrib"]].merge(
        mdf[["week_start", "total_spend"]], on="week_start"
    )

    monthly = merged.groupby("month").agg(
        avg_contrib=("total_contrib", "mean"),
        avg_spend=("total_spend", "mean"),
        n_weeks=("total_contrib", "count"),
    )
    monthly["efficiency"] = monthly["avg_contrib"] / (monthly["avg_spend"] + 1e-8)

    overall_eff = monthly["efficiency"].mean()
    monthly["index"] = monthly["efficiency"] / (overall_eff + 1e-8)

    indices = {}
    for m in range(1, 13):
        indices[m] = float(monthly.loc[m, "index"]) if m in monthly.index else 1.0

    return indices


def compute_event_boosts(
    results: MMMResults,
    model_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute data-driven efficiency multipliers for event types.

    Compares channel efficiency (contribution per SEK of spend) during
    event weeks vs baseline (non-event) weeks. Falls back to 1.0 if
    insufficient data.

    Returns:
        Dict {"heavy_discount": float, "light_discount": float, "product_drop": float}
    """
    defaults = {"heavy_discount": 1.0, "light_discount": 1.0, "product_drop": 1.0}

    if model_df is None or events_df is None or events_df.empty:
        return defaults

    try:
        contrib_df = results.channel_contributions.copy()
        contrib_df["week_start"] = pd.to_datetime(contrib_df["week_start"])
        channel_cols = [c for c in contrib_df.columns
                        if c not in ["week_start", "email"]]
        contrib_df["total_contrib"] = contrib_df[channel_cols].sum(axis=1)

        mdf = model_df.copy()
        mdf["week_start"] = pd.to_datetime(mdf["week_start"])
        spend_cols = [c for c in mdf.columns if c.endswith("_spend")]
        mdf["total_spend"] = mdf[spend_cols].sum(axis=1)

        edf = events_df.copy()
        edf["week_start"] = pd.to_datetime(edf["week_start"])

        merged = contrib_df[["week_start", "total_contrib"]].merge(
            mdf[["week_start", "total_spend"]], on="week_start"
        ).merge(
            edf[["week_start", "discount_campaign", "product_drop"]],
            on="week_start", how="left"
        )
        merged["discount_campaign"] = merged["discount_campaign"].fillna(0)
        merged["product_drop"] = merged["product_drop"].fillna(0)
        merged["efficiency"] = merged["total_contrib"] / (merged["total_spend"] + 1e-8)

        # Baseline: weeks with no events at all
        baseline_mask = (merged["discount_campaign"] == 0) & (merged["product_drop"] == 0)
        baseline_eff = merged.loc[baseline_mask, "efficiency"].mean()
        if baseline_eff <= 0 or not np.isfinite(baseline_eff):
            return defaults

        result = {}

        # Heavy discount (discount_campaign == 2)
        heavy = merged[merged["discount_campaign"] == 2]["efficiency"]
        result["heavy_discount"] = float(heavy.mean() / baseline_eff) if len(heavy) >= 2 else 1.0

        # Light discount (discount_campaign == 1)
        light = merged[merged["discount_campaign"] == 1]["efficiency"]
        result["light_discount"] = float(light.mean() / baseline_eff) if len(light) >= 2 else 1.0

        # Product drop
        drops = merged[merged["product_drop"] > 0]["efficiency"]
        result["product_drop"] = float(drops.mean() / baseline_eff) if len(drops) >= 2 else 1.0

        # Clamp to reasonable range (0.5 – 3.0) to avoid outliers
        for k in result:
            result[k] = max(0.5, min(3.0, result[k]))

        return result

    except Exception as e:
        logger.warning(f"Could not compute event boosts: {e}")
        return defaults


def compute_monthly_organic(
    results: MMMResults,
    model_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Get average organic (baseline) weekly revenue by calendar month.
    """
    baseline = results.baseline_contribution

    if model_df is not None:
        weeks = pd.to_datetime(model_df["week_start"])
        months = weeks.dt.month
    else:
        weeks = pd.date_range(
            results.date_range[0], periods=results.n_weeks, freq="W-MON"
        )
        months = weeks.month  # DatetimeIndex uses .month, not .dt.month

    df = pd.DataFrame({"month": months, "baseline": baseline})
    monthly_avg = df.groupby("month")["baseline"].mean()

    overall_avg = float(monthly_avg.mean())
    return {
        m: float(monthly_avg[m]) if m in monthly_avg.index else overall_avg
        for m in range(1, 13)
    }


# ═══════════════════════════════════════════════════════════════
# HISTORICAL BACKCHECK
# ═══════════════════════════════════════════════════════════════

def compute_historical_backcheck(
    results: MMMResults,
    model_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute historical monthly spend and aMER for validating projections.

    If the model projects spending 1.5M/month but you've never spent
    more than 500K, that's a red flag. This function provides the
    historical context to sanity-check recommendations.

    Returns DataFrame with month, total_spend, total_revenue, amer.
    """
    if model_df is None:
        return None

    try:
        mdf = model_df.copy()
        mdf["week_start"] = pd.to_datetime(mdf["week_start"])
        mdf["year_month"] = mdf["week_start"].dt.to_period("M")

        spend_cols = [c for c in mdf.columns if c.endswith("_spend")]
        mdf["total_spend"] = mdf[spend_cols].sum(axis=1)

        # Use new_revenue if available, otherwise total revenue
        rev_col = "new_revenue" if "new_revenue" in mdf.columns else "revenue"

        monthly = mdf.groupby("year_month").agg(
            total_spend=("total_spend", "sum"),
            total_revenue=(rev_col, "sum"),
            n_weeks=("total_spend", "count"),
        ).reset_index()

        monthly["amer"] = monthly["total_revenue"] / (monthly["total_spend"] + 1e-8)
        monthly["month_name"] = monthly["year_month"].astype(str)

        return monthly

    except Exception as e:
        logger.warning(f"Could not compute historical backcheck: {e}")
        return None


def compute_observed_yoy_trend(
    model_df: Optional[pd.DataFrame],
) -> Optional[dict]:
    """
    Compute the observed year-over-year trend in spend and efficiency
    from months where we have data for both this year and last year.

    For each overlapping calendar month, compares:
    - Spend change: did we spend more or less YoY?
    - Efficiency change: did aMER improve, decline, or stay flat?
    - Spend capacity: if aMER stayed ~flat, the spend change IS the
      growth in spend capacity.

    Returns dict with median YoY spend change, efficiency change,
    and per-month detail.
    """
    if model_df is None:
        return None

    try:
        mdf = model_df.copy()
        mdf["week_start"] = pd.to_datetime(mdf["week_start"])
        mdf["month"] = mdf["week_start"].dt.month
        mdf["year"] = mdf["week_start"].dt.year

        spend_cols = [c for c in mdf.columns if c.endswith("_spend")]
        mdf["total_spend"] = mdf[spend_cols].sum(axis=1)
        rev_col = "new_revenue" if "new_revenue" in mdf.columns else "revenue"

        # Aggregate to monthly
        monthly = mdf.groupby(["year", "month"]).agg(
            total_spend=("total_spend", "sum"),
            total_revenue=(rev_col, "sum"),
            n_weeks=("total_spend", "count"),
        ).reset_index()
        monthly["amer"] = monthly["total_revenue"] / (monthly["total_spend"] + 1e-8)

        years = sorted(monthly["year"].unique())
        if len(years) < 2:
            return None

        # Compare consecutive years for overlapping months
        comparisons = []
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            m1 = monthly[monthly["year"] == y1]
            m2 = monthly[monthly["year"] == y2]

            for _, row2 in m2.iterrows():
                m_num = row2["month"]
                row1 = m1[m1["month"] == m_num]
                if row1.empty:
                    continue
                row1 = row1.iloc[0]

                # Skip months with very low spend (noise)
                if row1["total_spend"] < 10000 or row2["total_spend"] < 10000:
                    continue
                # Skip partial months
                if row1["n_weeks"] < 3 or row2["n_weeks"] < 3:
                    continue

                spend_change_pct = (row2["total_spend"] - row1["total_spend"]) / (row1["total_spend"] + 1e-8) * 100
                amer_change_pct = (row2["amer"] - row1["amer"]) / (row1["amer"] + 1e-8) * 100

                comparisons.append({
                    "month": m_num,
                    "month_name": datetime.date(y2, m_num, 1).strftime("%b"),
                    "year_from": y1,
                    "year_to": y2,
                    "spend_y1": row1["total_spend"],
                    "spend_y2": row2["total_spend"],
                    "amer_y1": row1["amer"],
                    "amer_y2": row2["amer"],
                    "spend_change_pct": spend_change_pct,
                    "amer_change_pct": amer_change_pct,
                })

        if not comparisons:
            return None

        spend_changes = [c["spend_change_pct"] for c in comparisons]
        amer_changes = [c["amer_change_pct"] for c in comparisons]

        median_spend_change = float(np.median(spend_changes))
        median_amer_change = float(np.median(amer_changes))

        # "Spend capacity growth" = spend grew while aMER didn't collapse
        # If aMER dropped less than 10%, the spend growth is genuine capacity growth
        capacity_growth_months = [
            c for c in comparisons if c["amer_change_pct"] > -10
        ]
        if capacity_growth_months:
            observed_capacity_growth = float(np.median(
                [c["spend_change_pct"] for c in capacity_growth_months]
            ))
        else:
            observed_capacity_growth = 0.0

        return {
            "comparisons": comparisons,
            "n_overlapping_months": len(comparisons),
            "median_spend_change_pct": median_spend_change,
            "median_amer_change_pct": median_amer_change,
            "observed_capacity_growth_pct": observed_capacity_growth,
        }

    except Exception as e:
        logger.warning(f"Could not compute YoY trend: {e}")
        return None


def compute_same_month_benchmark(
    model_df: Optional[pd.DataFrame],
    target_month: int,
    target_year: int,
    yoy_growth_pct: float = 0.0,
) -> Optional[dict]:
    """
    Pull historical spend and aMER for the same calendar month from prior years.

    This is the simplest and most trustworthy sanity check:
    "Last May we spent 500K at 2.0x aMER. This May we can probably do 600K."

    Args:
        model_df: Historical weekly data with spend and revenue columns
        target_month: Calendar month number (1-12) we're planning for
        target_year: Year we're planning for
        yoy_growth_pct: Expected year-over-year growth in spend capacity

    Returns:
        Dict with prior year benchmarks and growth-adjusted suggestion,
        or None if insufficient data.
    """
    if model_df is None:
        return None

    try:
        mdf = model_df.copy()
        mdf["week_start"] = pd.to_datetime(mdf["week_start"])
        mdf["month"] = mdf["week_start"].dt.month
        mdf["year"] = mdf["week_start"].dt.year

        spend_cols = [c for c in mdf.columns if c.endswith("_spend")]
        mdf["total_spend"] = mdf[spend_cols].sum(axis=1)

        rev_col = "new_revenue" if "new_revenue" in mdf.columns else "revenue"

        # Per-channel spend for breakdown
        channel_spends = {}
        for col in spend_cols:
            ch_name = col.replace("_spend", "").replace("_", " ").title()
            channel_spends[ch_name] = col

        # Find same calendar month in prior years
        benchmarks = []
        for year in sorted(mdf["year"].unique()):
            if year >= target_year:
                continue
            month_data = mdf[(mdf["month"] == target_month) & (mdf["year"] == year)]
            if len(month_data) < 3:  # need at least 3 weeks
                continue

            total_spend = month_data["total_spend"].sum()
            total_rev = month_data[rev_col].sum()
            n_weeks = len(month_data)
            amer = total_rev / (total_spend + 1e-8)

            # Per-channel breakdown
            ch_breakdown = {}
            for ch_name, col in channel_spends.items():
                ch_breakdown[ch_name] = month_data[col].sum()

            benchmarks.append({
                "year": year,
                "month": target_month,
                "label": datetime.date(year, target_month, 1).strftime("%b %Y"),
                "total_spend": total_spend,
                "monthly_spend": total_spend,  # already monthly (sum of weeks)
                "total_revenue": total_rev,
                "amer": amer,
                "n_weeks": n_weeks,
                "channel_breakdown": ch_breakdown,
            })

        if not benchmarks:
            return None

        # Use most recent prior year as primary benchmark
        latest = benchmarks[-1]
        years_gap = target_year - latest["year"]
        growth_mult = (1 + yoy_growth_pct / 100) ** years_gap

        # Growth-adjusted suggestion: same aMER, more spend
        suggested_spend = latest["total_spend"] * growth_mult

        return {
            "benchmarks": benchmarks,
            "latest_benchmark": latest,
            "years_gap": years_gap,
            "growth_multiplier": growth_mult,
            "suggested_spend_same_amer": suggested_spend,
            "yoy_growth_pct": yoy_growth_pct,
        }

    except Exception as e:
        logger.warning(f"Could not compute same-month benchmark: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# MONTHLY SPEND PLAN
# ═══════════════════════════════════════════════════════════════

def monthly_spend_plan(
    results: MMMResults,
    gm2_pct: float,
    cltv_expansion_pct: float,
    seasonal_indices: dict,
    monthly_organic: dict,
    event_boosts: dict,
    months_ahead: int = 12,
    events_df: Optional[pd.DataFrame] = None,
    historical_max_monthly_spend: float = 0,
) -> pd.DataFrame:
    """
    Generate a month-by-month spend plan.

    Uses data-driven event boosts (from compute_event_boosts) rather than
    hardcoded multipliers. Flags months where recommended spend exceeds
    historical maximum.
    """
    today = datetime.date.today()
    rows = []

    heavy_mult = event_boosts.get("heavy_discount", 1.0)
    light_mult = event_boosts.get("light_discount", 1.0)
    drop_mult = event_boosts.get("product_drop", 1.0)

    for m in range(months_ahead):
        month_num = (today.month - 1 + m) % 12 + 1
        year = today.year + (today.month - 1 + m) // 12

        seasonal_mult = seasonal_indices.get(month_num, 1.0)
        organic_weekly = monthly_organic.get(month_num, 0)

        event_boost = 1.0
        has_heavy_discount = False
        has_light_discount = False
        has_product_drop = False

        if events_df is not None and not events_df.empty:
            edf = events_df.copy()
            edf["week_start"] = pd.to_datetime(edf["week_start"])
            month_events = edf[
                (edf["week_start"].dt.month == month_num) &
                (edf["week_start"].dt.year == year)
            ]
            if not month_events.empty:
                if "discount_campaign" in month_events.columns:
                    has_heavy_discount = (month_events["discount_campaign"] == 2).any()
                    has_light_discount = (
                        (month_events["discount_campaign"] == 1).any()
                        and not has_heavy_discount
                    )
                    if has_heavy_discount:
                        event_boost *= heavy_mult
                    elif has_light_discount:
                        event_boost *= light_mult
                if "product_drop" in month_events.columns:
                    has_product_drop = (month_events["product_drop"] > 0).any()
                    if has_product_drop:
                        event_boost *= drop_mult

        effective_mult = seasonal_mult * event_boost

        optimal = find_optimal_spend(
            results, gm2_pct, cltv_expansion_pct,
            organic_weekly_revenue=organic_weekly,
            seasonal_multiplier=effective_mult,
        )

        month_label = datetime.date(year, month_num, 1).strftime("%b %Y")
        events_str = []
        if has_heavy_discount:
            events_str.append(f"Heavy discount ({heavy_mult:.0%})")
        elif has_light_discount:
            events_str.append(f"Light discount ({light_mult:.0%})")
        if has_product_drop:
            events_str.append(f"Product drop ({drop_mult:.0%})")

        rec_monthly = optimal["optimal_monthly_spend"]
        exceeds_historical = (
            historical_max_monthly_spend > 0
            and rec_monthly > historical_max_monthly_spend * 1.5
        )

        rows.append({
            "month": f"{year}-{month_num:02d}",
            "month_name": month_label,
            "seasonal_index": round(seasonal_mult, 2),
            "event_boost": round(event_boost, 2),
            "effective_multiplier": round(effective_mult, 2),
            "recommended_weekly_spend": round(optimal["optimal_weekly_spend"], 0),
            "recommended_monthly_spend": round(rec_monthly, 0),
            "estimated_weekly_gp3_fo": round(optimal["optimal_gp3_first_order_weekly"], 0),
            "estimated_monthly_gp3_fo": round(optimal["optimal_gp3_first_order_monthly"], 0),
            "estimated_weekly_gp3_365d": round(optimal["optimal_gp3_365d_weekly"], 0),
            "estimated_monthly_gp3_365d": round(optimal["optimal_gp3_365d_monthly"], 0),
            "estimated_weekly_revenue": round(optimal["new_customer_revenue_weekly"], 0),
            "estimated_monthly_revenue": round(optimal["new_customer_revenue_monthly"], 0),
            "amer": round(optimal["amer_at_optimal"], 2),
            "events": ", ".join(events_str) if events_str else "—",
            "exceeds_historical": exceeds_historical,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# CHANNEL ALLOCATION AT A GIVEN SPEND LEVEL
# ═══════════════════════════════════════════════════════════════

def optimize_channel_allocation(
    results: MMMResults,
    total_weekly_spend: float,
    seasonal_multiplier: float = 1.0,
    calibration_factor: float = 1.0,
    min_pct: float = 0.05,
    max_pct: float = 0.80,
) -> pd.DataFrame:
    """
    Given a total spend level, find the optimal channel split.
    """
    params = results.channel_params
    roas_df = results.channel_roas
    channels = [ch for ch in params.keys() if ch != "email"]
    n_ch = len(channels)

    if total_weekly_spend <= 0 or n_ch == 0:
        return pd.DataFrame(columns=["channel", "weekly_spend", "monthly_spend",
                                      "pct", "weekly_revenue", "monthly_revenue"])

    adstock_means = {}
    current_spend = {}
    for ch in channels:
        row = roas_df[roas_df["channel"] == ch].iloc[0]
        adstock_means[ch] = _get_adstock_training_mean(params[ch], results.n_weeks, row)
        current_spend[ch] = row["total_spend"] / results.n_weeks

    current_total = sum(current_spend.values())
    eff_cal = _effective_calibration(calibration_factor, total_weekly_spend, current_total)

    def neg_revenue(allocation):
        total_rev = 0
        for i, ch in enumerate(channels):
            rev = predict_channel_revenue(allocation[i], params[ch], adstock_means[ch])
            total_rev += rev * eff_cal * seasonal_multiplier
        return -total_rev

    x0 = np.array([
        current_spend[ch] / (current_total + 1e-8) * total_weekly_spend
        for ch in channels
    ])

    constraints = [{"type": "eq", "fun": lambda x: sum(x) - total_weekly_spend}]
    bounds = [(total_weekly_spend * min_pct, total_weekly_spend * max_pct)] * n_ch

    result = minimize(
        neg_revenue, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
    )

    rows = []
    for i, ch in enumerate(channels):
        ch_spend = result.x[i]
        ch_rev = predict_channel_revenue(ch_spend, params[ch], adstock_means[ch])
        ch_rev *= eff_cal * seasonal_multiplier
        rows.append({
            "channel": ch,
            "weekly_spend": round(ch_spend, 0),
            "monthly_spend": round(ch_spend * 4.33, 0),
            "pct": round(ch_spend / (total_weekly_spend + 1e-8) * 100, 1),
            "weekly_revenue": round(ch_rev, 0),
            "monthly_revenue": round(ch_rev * 4.33, 0),
        })

    return pd.DataFrame(rows)
