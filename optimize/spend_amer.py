"""
Spend-aMER Model: Optimal spend planning for GP3 maximization.

Combines MMM saturation curves with unit economics (GM2%, 365D CLTV expansion)
to find the spend level that maximizes GP3 (gross profit after all variable
costs including marketing).

    GP3 = (New Customer Revenue × (1 + CLTV_expansion) × GM2%) − Marketing Spend

The breakeven aMER (acquisition Marketing Efficiency Ratio) is:

    breakeven_aMER = 1 / ((1 + CLTV_expansion) × GM2%)

At this aMER, each SEK of spend generates exactly enough contribution margin
to cover itself. The optimal spend is where marginal GP3 = 0, i.e., the point
on the saturation curve where spending one more SEK produces exactly one SEK
of contribution margin.

The model accounts for seasonal variation in channel efficiency and organic
demand to produce month-by-month spend recommendations.
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
    # Steady-state adstock ≈ spend / (1 - decay), discounted for warmup
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

    Args:
        spend_weekly: Constant weekly spend level (SEK)
        params: Channel parameters from MMMResults.channel_params
        adstock_training_mean: Mean of adstocked spend during training
        n_sim_weeks: Number of weeks to simulate (for adstock warmup)

    Returns:
        Predicted weekly revenue contribution from this channel
    """
    if spend_weekly <= 0:
        return 0.0

    spend_series = np.full(n_sim_weeks, spend_weekly)
    adstocked = geometric_adstock(spend_series, params["adstock_decay"], max_lag=8)

    # Normalize by TRAINING mean (not the new series mean)
    # This preserves absolute spend-level information
    x_norm = adstocked / (adstock_training_mean + 1e-8)

    # Hill saturation (manual — bypasses hill_saturation() which renormalizes)
    alpha = np.clip(params["saturation_alpha"], 0.01, 10.0)
    lam = np.clip(params["saturation_lam"], 0.01, 10.0)
    exponent = np.clip(lam * np.power(x_norm, alpha), 0, 30)
    saturated = 1 - np.exp(-exponent)

    # Average of last 4 weeks (after adstock has reached steady state)
    return params["beta_raw"] * saturated[-4:].mean()


# ═══════════════════════════════════════════════════════════════
# GP3 CURVE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_gp3_curve(
    results: MMMResults,
    gm2_pct: float,
    cltv_expansion_pct: float,
    organic_weekly_revenue: float = 0,
    seasonal_multiplier: float = 1.0,
    n_points: int = 150,
    max_spend_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Compute GP3 at various total spend levels.

    This generates the "GP3 parabola" — the concave curve showing how GP3
    rises (as spend generates revenue) then falls (as saturation causes
    diminishing returns that can't cover the marginal cost of spend).

    Returns DataFrame with columns:
        weekly_spend, monthly_spend, paid_new_customer_revenue,
        total_new_customer_revenue, revenue_365d, contribution_margin,
        gp3, gp3_monthly, amer, marginal_roas
    """
    params = results.channel_params
    roas_df = results.channel_roas
    channels = [ch for ch in params.keys() if ch != "email"]

    # Current weekly spend and adstock means per channel
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

        channel_rev *= seasonal_multiplier

        total_new_rev = organic_weekly_revenue + channel_rev
        rev_365d = total_new_rev * cltv_mult
        cm = rev_365d * gm2_frac
        gp3 = cm - total_spend
        amer = total_new_rev / (total_spend + 1e-8) if total_spend > 0 else 0

        rows.append({
            "weekly_spend": total_spend,
            "monthly_spend": total_spend * 4.33,
            "paid_new_customer_revenue": channel_rev,
            "total_new_customer_revenue": total_new_rev,
            "revenue_365d": rev_365d,
            "contribution_margin": cm,
            "gp3": gp3,
            "gp3_monthly": gp3 * 4.33,
            "amer": amer,
        })

    df = pd.DataFrame(rows)

    # Marginal ROAS = d(paid_revenue) / d(spend)
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
    max_spend_mult: float = 5.0,
) -> dict:
    """
    Find the weekly spend level that maximizes GP3.

    The optimum is where marginal channel revenue × CLTV × GM2% = 1,
    i.e., the last SEK of spend produces exactly 1 SEK of contribution margin.

    Returns:
        Dict with optimal_weekly_spend, optimal_monthly_spend,
        optimal_gp3_weekly/monthly, new_customer_revenue, amer,
        breakeven_amer, channel_allocation, comparison to current.
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
    breakeven_amer = 1 / (cltv_mult * gm2_frac)

    def neg_gp3(total_spend):
        if total_spend <= 0:
            return -(organic_weekly_revenue * cltv_mult * gm2_frac)
        channel_rev = 0
        for ch in channels:
            ch_spend = total_spend * alloc_ratios.get(ch, 1 / len(channels))
            ch_rev = predict_channel_revenue(ch_spend, params[ch], adstock_means[ch])
            channel_rev += ch_rev
        channel_rev *= seasonal_multiplier
        total_new_rev = organic_weekly_revenue + channel_rev
        gp3 = total_new_rev * cltv_mult * gm2_frac - total_spend
        return -gp3

    result = minimize_scalar(
        neg_gp3,
        bounds=(0, current_total * max_spend_mult),
        method="bounded",
    )

    optimal_spend = result.x
    optimal_gp3 = -result.fun

    # Channel breakdown at optimal
    total_channel_rev = 0
    channel_allocation = {}
    for ch in channels:
        ch_spend = optimal_spend * alloc_ratios.get(ch, 1 / len(channels))
        ch_rev = predict_channel_revenue(ch_spend, params[ch], adstock_means[ch]) * seasonal_multiplier
        channel_allocation[ch] = {
            "weekly_spend": ch_spend,
            "monthly_spend": ch_spend * 4.33,
            "weekly_revenue": ch_rev,
        }
        total_channel_rev += ch_rev

    total_new_rev = organic_weekly_revenue + total_channel_rev

    # Current GP3 for comparison
    current_channel_rev = 0
    for ch in channels:
        ch_rev = predict_channel_revenue(current_spend[ch], params[ch], adstock_means[ch]) * seasonal_multiplier
        current_channel_rev += ch_rev
    current_total_rev = organic_weekly_revenue + current_channel_rev
    current_gp3 = current_total_rev * cltv_mult * gm2_frac - current_total

    return {
        "optimal_weekly_spend": optimal_spend,
        "optimal_monthly_spend": optimal_spend * 4.33,
        "optimal_gp3_weekly": optimal_gp3,
        "optimal_gp3_monthly": optimal_gp3 * 4.33,
        "new_customer_revenue_weekly": total_new_rev,
        "new_customer_revenue_monthly": total_new_rev * 4.33,
        "amer_at_optimal": total_new_rev / (optimal_spend + 1e-8),
        "breakeven_amer": breakeven_amer,
        "channel_allocation": channel_allocation,
        "current_weekly_spend": current_total,
        "current_monthly_spend": current_total * 4.33,
        "current_gp3_weekly": current_gp3,
        "current_gp3_monthly": current_gp3 * 4.33,
        "spend_change_pct": (optimal_spend - current_total) / (current_total + 1e-8) * 100,
    }


# ═══════════════════════════════════════════════════════════════
# SEASONAL INDICES
# ═══════════════════════════════════════════════════════════════

def compute_seasonal_indices(
    results: MMMResults,
    model_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute monthly seasonal efficiency indices from historical data.

    For each calendar month, computes the ratio of channel efficiency
    (contribution per SEK of spend) vs the overall average.

    November might be 1.3 (30% more efficient per SEK),
    January might be 0.8 (20% less efficient).

    Returns:
        Dict {month_number: efficiency_index} where 1.0 = average month
    """
    if model_df is None:
        return {m: 1.0 for m in range(1, 13)}

    contrib_df = results.channel_contributions.copy()
    contrib_df["week_start"] = pd.to_datetime(contrib_df["week_start"])
    contrib_df["month"] = contrib_df["week_start"].dt.month

    # Channel contribution columns (exclude email, metadata)
    channel_cols = [c for c in contrib_df.columns
                    if c not in ["week_start", "month", "email"]]
    if not channel_cols:
        return {m: 1.0 for m in range(1, 13)}

    contrib_df["total_contrib"] = contrib_df[channel_cols].sum(axis=1)

    # Get weekly spend from model_df
    mdf = model_df.copy()
    mdf["week_start"] = pd.to_datetime(mdf["week_start"])
    mdf["month"] = mdf["week_start"].dt.month
    spend_cols = [c for c in mdf.columns if c.endswith("_spend")]
    mdf["total_spend"] = mdf[spend_cols].sum(axis=1)

    merged = contrib_df[["week_start", "month", "total_contrib"]].merge(
        mdf[["week_start", "total_spend"]], on="week_start"
    )

    # Monthly avg efficiency = avg(contribution) / avg(spend)
    monthly = merged.groupby("month").agg(
        avg_contrib=("total_contrib", "mean"),
        avg_spend=("total_spend", "mean"),
        n_weeks=("total_contrib", "count"),
    )
    monthly["efficiency"] = monthly["avg_contrib"] / (monthly["avg_spend"] + 1e-8)

    # Normalize so mean = 1.0
    overall_eff = monthly["efficiency"].mean()
    monthly["index"] = monthly["efficiency"] / (overall_eff + 1e-8)

    indices = {}
    for m in range(1, 13):
        indices[m] = float(monthly.loc[m, "index"]) if m in monthly.index else 1.0

    return indices


def compute_monthly_organic(
    results: MMMResults,
    model_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Get average organic (baseline) weekly revenue by calendar month.

    Returns:
        Dict {month_number: avg_weekly_organic_revenue}
    """
    baseline = results.baseline_contribution

    if model_df is not None:
        weeks = pd.to_datetime(model_df["week_start"])
    else:
        weeks = pd.date_range(
            results.date_range[0], periods=results.n_weeks, freq="W-MON"
        )

    df = pd.DataFrame({"month": weeks.dt.month, "baseline": baseline})
    monthly_avg = df.groupby("month")["baseline"].mean()

    overall_avg = float(monthly_avg.mean())
    return {
        m: float(monthly_avg[m]) if m in monthly_avg.index else overall_avg
        for m in range(1, 13)
    }


# ═══════════════════════════════════════════════════════════════
# MONTHLY SPEND PLAN
# ═══════════════════════════════════════════════════════════════

def monthly_spend_plan(
    results: MMMResults,
    gm2_pct: float,
    cltv_expansion_pct: float,
    seasonal_indices: dict,
    monthly_organic: dict,
    months_ahead: int = 12,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate a month-by-month spend plan.

    For each forward month:
    1. Look up seasonal efficiency index for that calendar month
    2. Check for planned events (discount campaigns → efficiency boost)
    3. Find optimal spend given the effective multiplier
    4. Project GP3 and aMER

    Returns DataFrame with one row per month.
    """
    today = datetime.date.today()
    rows = []

    for m in range(months_ahead):
        month_num = (today.month - 1 + m) % 12 + 1
        year = today.year + (today.month - 1 + m) // 12

        seasonal_mult = seasonal_indices.get(month_num, 1.0)
        organic_weekly = monthly_organic.get(month_num, 0)

        # Event boost: planned discount campaigns and product drops
        # increase channel efficiency (better conversion during events)
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
                    has_light_discount = (month_events["discount_campaign"] == 1).any()
                    if has_heavy_discount:
                        event_boost *= 1.30
                    elif has_light_discount:
                        event_boost *= 1.15
                if "product_drop" in month_events.columns:
                    has_product_drop = (month_events["product_drop"] > 0).any()
                    if has_product_drop:
                        event_boost *= 1.10

        effective_mult = seasonal_mult * event_boost

        # Find optimal for this month's conditions
        optimal = find_optimal_spend(
            results, gm2_pct, cltv_expansion_pct,
            organic_weekly_revenue=organic_weekly,
            seasonal_multiplier=effective_mult,
        )

        month_label = datetime.date(year, month_num, 1).strftime("%b %Y")
        events_str = []
        if has_heavy_discount:
            events_str.append("Heavy discount")
        elif has_light_discount:
            events_str.append("Light discount")
        if has_product_drop:
            events_str.append("Product drop")

        rows.append({
            "month": f"{year}-{month_num:02d}",
            "month_name": month_label,
            "seasonal_index": round(seasonal_mult, 2),
            "event_boost": round(event_boost, 2),
            "effective_multiplier": round(effective_mult, 2),
            "recommended_weekly_spend": round(optimal["optimal_weekly_spend"], 0),
            "recommended_monthly_spend": round(optimal["optimal_monthly_spend"], 0),
            "estimated_weekly_gp3": round(optimal["optimal_gp3_weekly"], 0),
            "estimated_monthly_gp3": round(optimal["optimal_gp3_monthly"], 0),
            "estimated_weekly_revenue": round(optimal["new_customer_revenue_weekly"], 0),
            "estimated_monthly_revenue": round(optimal["new_customer_revenue_monthly"], 0),
            "amer": round(optimal["amer_at_optimal"], 2),
            "events": ", ".join(events_str) if events_str else "—",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# CHANNEL ALLOCATION AT A GIVEN SPEND LEVEL
# ═══════════════════════════════════════════════════════════════

def optimize_channel_allocation(
    results: MMMResults,
    total_weekly_spend: float,
    seasonal_multiplier: float = 1.0,
    min_pct: float = 0.05,
    max_pct: float = 0.80,
) -> pd.DataFrame:
    """
    Given a total spend level, find the optimal channel split.

    Uses SLSQP to maximize total channel revenue subject to
    per-channel min/max constraints.
    """
    params = results.channel_params
    roas_df = results.channel_roas
    channels = [ch for ch in params.keys() if ch != "email"]
    n_ch = len(channels)

    adstock_means = {}
    current_spend = {}
    for ch in channels:
        row = roas_df[roas_df["channel"] == ch].iloc[0]
        adstock_means[ch] = _get_adstock_training_mean(params[ch], results.n_weeks, row)
        current_spend[ch] = row["total_spend"] / results.n_weeks

    current_total = sum(current_spend.values())

    def neg_revenue(allocation):
        total_rev = 0
        for i, ch in enumerate(channels):
            rev = predict_channel_revenue(allocation[i], params[ch], adstock_means[ch])
            total_rev += rev * seasonal_multiplier
        return -total_rev

    # Initialize proportionally to current allocation
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
        ch_rev *= seasonal_multiplier
        rows.append({
            "channel": ch,
            "weekly_spend": round(ch_spend, 0),
            "monthly_spend": round(ch_spend * 4.33, 0),
            "pct": round(ch_spend / (total_weekly_spend + 1e-8) * 100, 1),
            "weekly_revenue": round(ch_rev, 0),
            "monthly_revenue": round(ch_rev * 4.33, 0),
        })

    return pd.DataFrame(rows)
