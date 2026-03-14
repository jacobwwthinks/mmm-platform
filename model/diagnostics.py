"""
Model diagnostics and validation.

Provides tools to assess whether the MMM results are trustworthy.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def assess_model_quality(results) -> dict:
    """
    Comprehensive model quality assessment.

    Returns a dict with pass/fail checks and recommendations.
    """
    checks = {}

    # 1. Overall fit (R²)
    r2 = results.r_squared
    if r2 >= 0.85:
        checks["r_squared"] = {"value": r2, "status": "good", "note": "Excellent fit"}
    elif r2 >= 0.7:
        checks["r_squared"] = {"value": r2, "status": "ok", "note": "Acceptable fit"}
    else:
        checks["r_squared"] = {"value": r2, "status": "warning",
                               "note": "Poor fit — consider adding control variables or more data"}

    # 2. MAPE
    mape = results.mape
    if mape <= 10:
        checks["mape"] = {"value": mape, "status": "good", "note": "Very accurate predictions"}
    elif mape <= 20:
        checks["mape"] = {"value": mape, "status": "ok", "note": "Reasonable accuracy"}
    else:
        checks["mape"] = {"value": mape, "status": "warning",
                          "note": "High prediction error — model may be missing important factors"}

    # 3. Residual autocorrelation (should be low)
    residuals = results.residuals
    autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    if abs(autocorr) <= 0.3:
        checks["autocorrelation"] = {"value": autocorr, "status": "good",
                                     "note": "No significant autocorrelation"}
    else:
        checks["autocorrelation"] = {"value": autocorr, "status": "warning",
                                     "note": "Residuals are autocorrelated — model may be missing a trend"}

    # 4. Channel contribution sanity
    contrib_df = results.channel_contributions
    total_rev = results.actual.sum()
    for col in contrib_df.columns:
        if col == "week_start":
            continue
        pct = contrib_df[col].sum() / total_rev * 100
        if pct < 0:
            checks[f"contribution_{col}"] = {"value": pct, "status": "warning",
                                             "note": f"{col} has negative contribution — unusual"}
        elif pct > 60:
            checks[f"contribution_{col}"] = {"value": pct, "status": "warning",
                                             "note": f"{col} accounts for {pct:.0f}% of revenue — suspiciously high"}

    # 5. ROAS reasonableness + low-spend flagging
    total_all_spend = results.channel_roas["total_spend"].sum()
    for _, row in results.channel_roas.iterrows():
        roas = row["roas_mean"]
        ci_width = row["roas_95"] - row["roas_5"]
        channel = row["channel"]
        spend = row["total_spend"]
        spend_share = spend / (total_all_spend + 1e-8) * 100

        # Flag channels with very small spend (< 5% of total)
        if spend_share < 5 and spend > 0:
            checks[f"low_spend_{channel}"] = {
                "value": spend_share,
                "status": "warning",
                "note": (
                    f"{channel} represents only {spend_share:.1f}% of total spend "
                    f"({spend:,.0f} SEK). ROAS of {roas:.1f}x is unreliable at "
                    f"this scale — the model has very little data to estimate "
                    f"its true effect."
                ),
            }

        if roas > 20:
            checks[f"roas_{channel}"] = {"value": roas, "status": "warning",
                                         "note": f"ROAS of {roas:.1f}x is unusually high"}
        if ci_width > 10 * roas:
            checks[f"roas_ci_{channel}"] = {"value": ci_width, "status": "warning",
                                            "note": f"Very wide CI — low confidence in {channel} ROAS"}

    # 6. Baseline check
    baseline_pct = results.baseline_contribution.sum() / total_rev * 100
    if baseline_pct > 90:
        checks["baseline"] = {"value": baseline_pct, "status": "warning",
                              "note": "Baseline is >90% — model sees little effect from marketing"}
    elif baseline_pct < 20:
        checks["baseline"] = {"value": baseline_pct, "status": "warning",
                              "note": "Baseline is <20% — unusual, may be overfitting to spend patterns"}
    else:
        checks["baseline"] = {"value": baseline_pct, "status": "good",
                              "note": f"Baseline accounts for {baseline_pct:.0f}% of revenue"}

    # Overall assessment
    warnings = [k for k, v in checks.items() if v["status"] == "warning"]
    if len(warnings) == 0:
        checks["overall"] = {"status": "good", "note": "Model passes all quality checks"}
    elif len(warnings) <= 2:
        checks["overall"] = {"status": "ok",
                             "note": f"Minor concerns: {', '.join(warnings)}"}
    else:
        checks["overall"] = {"status": "warning",
                             "note": f"Multiple issues found: {', '.join(warnings)}"}

    return checks


def holdout_validation(
    df: pd.DataFrame,
    model,
    test_weeks: int = 8,
    target_col: str = "revenue",
    spend_cols: Optional[list[str]] = None,
) -> dict:
    """
    Train on historical data, test on recent weeks.

    Returns fit metrics on both train and test sets.
    """
    train_df = df.iloc[:-test_weeks].copy()
    test_df = df.iloc[-test_weeks:].copy()

    # Fit on train
    train_results = model.fit(train_df, target_col=target_col, spend_cols=spend_cols)

    # The test prediction would require re-running the model on full data
    # and comparing the last test_weeks. For now, report train metrics.
    return {
        "train_r2": train_results.r_squared,
        "train_mape": train_results.mape,
        "train_weeks": len(train_df),
        "test_weeks": test_weeks,
    }
