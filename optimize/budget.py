"""
Budget optimization based on fitted MMM results.

Given the model's understanding of channel saturation curves,
recommends how to reallocate budget for maximum revenue.
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from typing import Optional

from model.mmm import geometric_adstock, hill_saturation

logger = logging.getLogger(__name__)


def optimize_budget(
    results,
    total_budget: Optional[float] = None,
    min_spend_pct: float = 0.05,
    max_spend_pct: float = 0.80,
) -> pd.DataFrame:
    """
    Find the optimal budget allocation across channels.

    Uses the fitted saturation curves to determine which channels
    have the most room for additional spend (not yet saturated)
    and which are oversaturated.

    Args:
        results: MMMResults from model fitting
        total_budget: Total weekly budget to allocate (default: current total)
        min_spend_pct: Minimum % of budget per channel (prevents zeroing out)
        max_spend_pct: Maximum % of budget per channel
    Returns:
        DataFrame with current vs. recommended allocation
    """
    roas_df = results.channel_roas
    params = results.channel_params
    contrib_df = results.channel_contributions
    n_weeks = results.n_weeks

    channels = list(params.keys())
    n_channels = len(channels)

    # Current weekly spend per channel
    current_weekly_spend = {}
    for ch in channels:
        col = f"{ch}_spend" if f"{ch}_spend" in (results.spend_columns or []) else ch
        current_weekly_spend[ch] = roas_df[roas_df["channel"] == ch]["total_spend"].values[0] / n_weeks

    current_total = sum(current_weekly_spend.values())
    if total_budget is None:
        total_budget = current_total

    def predicted_revenue(allocation):
        """Predict total channel revenue given weekly spend allocation."""
        total_rev = 0
        for i, ch in enumerate(channels):
            p = params[ch]
            weekly_spend = np.full(13, allocation[i])  # simulate 13 weeks
            adstocked = geometric_adstock(weekly_spend, p["adstock_decay"], max_lag=8)
            saturated = hill_saturation(adstocked, p["saturation_alpha"], p["saturation_lam"])
            total_rev += p["beta"] * saturated.mean()  # average weekly contribution
        return total_rev

    def neg_revenue(allocation):
        return -predicted_revenue(allocation)

    # Constraints: total spend = budget
    constraints = [{"type": "eq", "fun": lambda x: sum(x) - total_budget}]

    # Bounds: each channel between min and max of total budget
    bounds = [(total_budget * min_spend_pct, total_budget * max_spend_pct)] * n_channels

    # Initialize with proportional allocation
    x0 = np.array([current_weekly_spend.get(ch, total_budget / n_channels) for ch in channels])
    x0 = x0 * total_budget / (x0.sum() + 1e-8)  # normalize to budget

    result = minimize(
        neg_revenue,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    optimal_allocation = result.x

    # Build results table
    rows = []
    for i, ch in enumerate(channels):
        current = current_weekly_spend.get(ch, 0)
        optimal = optimal_allocation[i]
        change_pct = (optimal - current) / (current + 1e-8) * 100

        rows.append({
            "channel": ch,
            "current_weekly_spend": round(current, 2),
            "current_pct": round(current / current_total * 100, 1) if current_total > 0 else 0,
            "recommended_weekly_spend": round(optimal, 2),
            "recommended_pct": round(optimal / total_budget * 100, 1),
            "change_pct": round(change_pct, 1),
            "direction": "increase" if change_pct > 5 else ("decrease" if change_pct < -5 else "maintain"),
        })

    opt_df = pd.DataFrame(rows)

    # Estimate revenue impact
    current_rev = predicted_revenue(np.array([current_weekly_spend[ch] for ch in channels]))
    optimal_rev = predicted_revenue(optimal_allocation)
    lift_pct = (optimal_rev - current_rev) / (current_rev + 1e-8) * 100

    logger.info(f"Budget optimization complete:")
    logger.info(f"  Total budget: {total_budget:,.0f}/week")
    logger.info(f"  Estimated revenue lift: {lift_pct:+.1f}%")

    opt_df.attrs["total_budget"] = total_budget
    opt_df.attrs["estimated_lift_pct"] = round(lift_pct, 1)
    opt_df.attrs["current_rev_weekly"] = round(current_rev, 2)
    opt_df.attrs["optimal_rev_weekly"] = round(optimal_rev, 2)

    return opt_df


def scenario_analysis(
    results,
    budget_multipliers: list[float] = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
) -> pd.DataFrame:
    """
    Run multiple budget scenarios to show diminishing returns at portfolio level.

    Args:
        results: Fitted MMMResults
        budget_multipliers: List of multipliers (1.0 = current budget)

    Returns:
        DataFrame with budget level and expected revenue for each scenario
    """
    params = results.channel_params
    channels = list(params.keys())
    roas_df = results.channel_roas
    n_weeks = results.n_weeks

    current_weekly_spend = {}
    for ch in channels:
        current_weekly_spend[ch] = roas_df[roas_df["channel"] == ch]["total_spend"].values[0] / n_weeks
    current_total = sum(current_weekly_spend.values())

    scenarios = []
    for mult in budget_multipliers:
        budget = current_total * mult

        # Optimize for this budget level
        opt_df = optimize_budget(results, total_budget=budget)

        scenarios.append({
            "budget_multiplier": mult,
            "weekly_budget": round(budget, 0),
            "estimated_weekly_revenue": opt_df.attrs.get("optimal_rev_weekly", 0),
            "estimated_lift_vs_current": opt_df.attrs.get("estimated_lift_pct", 0),
        })

    return pd.DataFrame(scenarios)
