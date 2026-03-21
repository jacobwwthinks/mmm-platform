"""
Marketing Mix Model using Bayesian inference.

This module wraps the MMM modeling process with a clean API.
It supports Google's Meridian as the primary backend, with a
lightweight PyMC-based fallback.

The model decomposes revenue into:
    revenue = baseline + trend + seasonality + sum(channel_effects) + sum(control_effects) + noise

Where each channel_effect = effective_beta * saturation(adstock(spend), effective_lam)

Two types of time-varying channel effectiveness:

1. **beta_trend** (vertical shift): The same saturated spend produces more
   revenue over time (e.g., creative quality improvements). This captures
   "same spend → more revenue."

2. **lam_trend** (horizontal shift): The saturation curve itself shifts —
   the channel can absorb more spend before hitting diminishing returns.
   This captures "can spend MORE at the same ROAS." A negative lam_trend
   means the half-saturation point moves higher over time (growing capacity).

The channel × event interaction terms (beta_event) allow each channel's
effectiveness to scale up during heavy discount / Black Week periods.

All internal computations happen in z-score (normalized) space so that
the channel beta priors (~0.1-0.5) represent fractions of revenue's
standard deviation, not raw currency units.
"""

import numpy as np
import pandas as pd
import logging
import json
import pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from model.priors import get_channel_prior, get_spillover_pairs, CONTROL_PRIORS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ADSTOCK & SATURATION TRANSFORMS
# ═══════════════════════════════════════════════════════════════

def geometric_adstock(x: np.ndarray, decay: float, max_lag: int = 8) -> np.ndarray:
    """
    Apply geometric adstock transformation.

    Simulates the carryover effect: an ad seen this week still has
    a decaying effect in subsequent weeks.

    Uses a single-pass recursive formula: adstock[t] = x[t] + decay * adstock[t-1]
    which is equivalent to the lagged sum but O(T) instead of O(T * max_lag).

    Args:
        x: Spend time series (T,)
        decay: Decay rate 0-1 (higher = longer memory)
        max_lag: Maximum number of weeks to carry over (kept for API compat)

    Returns:
        Transformed series with carryover effects
    """
    T = len(x)
    result = np.empty(T)
    result[0] = x[0]
    for t in range(1, T):
        result[t] = x[t] + decay * result[t - 1]
    return result


def hill_saturation(x: np.ndarray, alpha: float, lam: float) -> np.ndarray:
    """
    Hill function for diminishing returns.

    As spend increases, the incremental effect decreases.
    This is the "S-curve" that captures saturation.

    Args:
        x: Adstocked spend (T,)
        alpha: Shape/steepness (higher = sharper saturation)
        lam: Scale (higher = saturation kicks in later)

    Returns:
        Saturated spend effect (0-1 range)
    """
    # Normalize x
    x_norm = x / (x.mean() + 1e-8)
    # Clip to prevent overflow in power/exp
    alpha = np.clip(alpha, 0.01, 10.0)
    lam = np.clip(lam, 0.01, 10.0)
    exponent = np.clip(lam * np.power(x_norm, alpha), 0, 30)
    return 1 - np.exp(-exponent)


def _safe_exp(x, max_val=20.0):
    """Numerically safe exp with clipping."""
    return np.exp(np.clip(x, -max_val, max_val))


def _safe_sigmoid(x):
    """Numerically safe sigmoid."""
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))


# ═══════════════════════════════════════════════════════════════
# RESULTS CONTAINER
# ═══════════════════════════════════════════════════════════════

@dataclass
class MMMResults:
    """Container for MMM fitting results."""

    # Model parameters (posterior means + credible intervals)
    channel_contributions: pd.DataFrame    # Weekly contribution per channel
    channel_roas: pd.DataFrame             # ROAS per channel with CI
    channel_params: dict                   # Fitted adstock, saturation params
    baseline_contribution: np.ndarray      # Baseline (organic) revenue
    control_contributions: pd.DataFrame    # Effect of promos, drops, holidays
    seasonality: np.ndarray                # Seasonal component

    # Model fit quality
    actual: np.ndarray                     # Actual revenue
    predicted: np.ndarray                  # Model predicted revenue
    residuals: np.ndarray                  # Actual - predicted
    r_squared: float                       # Overall fit
    mape: float                            # Mean absolute % error

    # Raw posterior samples (for uncertainty quantification)
    posterior_samples: Optional[dict] = None

    # Metadata
    spend_columns: list = None
    date_range: tuple = None
    n_weeks: int = 0
    target_col: str = "revenue"            # What the model was fit on

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"MMM Results Summary",
            f"{'='*50}",
            f"Period: {self.date_range[0]} to {self.date_range[1]} ({self.n_weeks} weeks)",
            f"Model fit: R² = {self.r_squared:.3f}, MAPE = {self.mape:.1f}%",
            f"",
            f"Channel ROAS (posterior mean [90% CI]):",
        ]
        for _, row in self.channel_roas.iterrows():
            lines.append(
                f"  {row['channel']:20s}: {row['roas_mean']:.2f}x "
                f"[{row['roas_5']:.2f} - {row['roas_95']:.2f}]"
            )
        lines.append(f"")
        lines.append(f"Revenue Decomposition:")
        total_rev = self.actual.sum()
        baseline_pct = self.baseline_contribution.sum() / total_rev * 100
        lines.append(f"  {'Baseline (organic)':20s}: {baseline_pct:.1f}%")
        for col in self.channel_contributions.columns:
            if col == "week_start":
                continue
            pct = self.channel_contributions[col].sum() / total_rev * 100
            lines.append(f"  {col:20s}: {pct:.1f}%")
        return "\n".join(lines)

    def save(self, path: str, model_df: Optional[pd.DataFrame] = None):
        """Save results to disk, optionally including the model DataFrame."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.channel_contributions.to_csv(f"{path}/channel_contributions.csv", index=False)
        self.channel_roas.to_csv(f"{path}/channel_roas.csv", index=False)
        with open(f"{path}/params.json", "w") as f:
            json.dump(self.channel_params, f, indent=2, default=str)
        with open(f"{path}/results.pkl", "wb") as f:
            pickle.dump(self, f)
        if model_df is not None:
            with open(f"{path}/model_df.pkl", "wb") as f:
                pickle.dump(model_df, f)
            logger.info(f"Model DataFrame saved ({len(model_df)} rows)")

            # Auto-train aMER model from the same data
            try:
                from optimize.amer_model import fit_amer_model, save_amer_coefficients
                coefficients = fit_amer_model(model_df)
                save_amer_coefficients(coefficients, path)
                logger.info(
                    f"aMER model trained: R²={coefficients['r_squared']:.3f}, "
                    f"{coefficients['n_observations']} months"
                )
            except Exception as e:
                logger.warning(f"Could not auto-train aMER model: {e}")

        logger.info(f"Results saved to {path}/")

    @classmethod
    def load(cls, path: str) -> "MMMResults":
        """Load results from disk. Returns None if file is missing or corrupt."""
        pkl_path = f"{path}/results.pkl"
        try:
            with open(pkl_path, "rb") as f:
                results = pickle.load(f)
            # Backcompat: add target_col if missing (old pickles)
            if not hasattr(results, "target_col"):
                results.target_col = "revenue"
            return results
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            logger.warning(f"Could not load results from {pkl_path}: {e}")
            # Remove corrupt file so we don't keep hitting this
            import os
            try:
                os.remove(pkl_path)
                logger.info(f"Removed corrupt results file: {pkl_path}")
            except OSError:
                pass
            return None

    @classmethod
    def load_model_df(cls, path: str) -> Optional[pd.DataFrame]:
        """Load persisted model_df from disk. Returns None if not available."""
        df_path = f"{path}/model_df.pkl"
        try:
            with open(df_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════
# LIGHTWEIGHT BAYESIAN MMM (built-in, no external framework needed)
# ═══════════════════════════════════════════════════════════════

class LightweightMMM:
    """
    Bayesian Marketing Mix Model using NumPyro/JAX or scipy optimization.

    This is a self-contained implementation that doesn't require Meridian
    or PyMC-Marketing, making deployment simpler. It implements the same
    core model: revenue = baseline + seasonality + adstock(saturation(spend)) + controls

    For production use with Meridian, see MeridianMMM class below.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.n_warmup = self.config.get("n_warmup", 500)
        self.n_samples = self.config.get("n_samples", 500)
        self.n_chains = self.config.get("n_chains", 2)

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "revenue",
        spend_cols: Optional[list[str]] = None,
        control_cols: Optional[list[str]] = None,
    ) -> MMMResults:
        """
        Fit the MMM model.

        Uses MAP estimation with bootstrap uncertainty quantification.
        All internal computations happen in z-score (normalized) space
        so that channel beta priors (~0.1-0.5) represent meaningful
        fractions of revenue variation.

        Args:
            df: Model-ready DataFrame with week_start, target, spend, and control columns
            target_col: Name of the revenue/KPI column
            spend_cols: List of spend column names (auto-detected if None)
            control_cols: List of control variable columns

        Returns:
            MMMResults with decomposition and ROAS estimates
        """
        from scipy.optimize import minimize

        # Auto-detect columns
        if spend_cols is None:
            spend_cols = sorted([c for c in df.columns if c.endswith("_spend") and df[c].sum() > 0])
            if "email_opens" in df.columns and df["email_opens"].sum() > 0:
                spend_cols.append("email_opens")
        if control_cols is None:
            control_cols = [c for c in ["discount_campaign", "product_drop", "product_offering", "holiday"]
                          if c in df.columns]

        y = df[target_col].values.astype(float)
        T = len(y)
        n_channels = len(spend_cols)

        # Normalize target to z-scores for stable optimization
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y_norm = (y - y_mean) / y_std

        # Prepare spend matrices
        spend_matrix = np.column_stack([df[col].values.astype(float) for col in spend_cols])

        # Heavy discount indicator for channel × event interactions
        # During heavy discount weeks (Black Week etc), conversion rates surge —
        # the same ad spend generates much more revenue. This interaction term
        # lets each channel's beta scale up during these periods.
        heavy_discount = np.zeros(T)
        if "discount_campaign" in df.columns:
            heavy_discount = (df["discount_campaign"].values >= 2).astype(float)
        n_heavy_weeks = int(heavy_discount.sum())
        has_event_interactions = n_heavy_weeks >= 2  # need at least 2 weeks to fit

        # When event interactions are active, REMOVE discount_campaign from the
        # additive controls — the interaction terms model the discount effect
        # multiplicatively (per-channel), which is more accurate. Keeping both
        # confuses the optimizer: the additive control steals variance from the
        # interaction terms, weakening the event boost.
        if has_event_interactions and "discount_campaign" in control_cols:
            control_cols = [c for c in control_cols if c != "discount_campaign"]
            logger.info("Removed discount_campaign from controls (handled by channel × event interactions)")

        control_matrix = np.column_stack([df[col].values.astype(float) for col in control_cols]) \
            if control_cols else np.zeros((T, 0))

        logger.info(f"Heavy discount weeks: {n_heavy_weeks} — "
                    f"event interactions {'enabled' if has_event_interactions else 'disabled (too few weeks)'}")

        # Time index for linear trend (normalized 0 to 1)
        time_index = np.linspace(0, 1, T)

        # Seasonality features (Fourier)
        # Use 5 harmonics (10 features) to capture sharper seasonal patterns
        # like Black Week spikes that 3 harmonics (smooth curves) would miss
        week_of_year = df["week_start"].dt.isocalendar().week.values.astype(float)
        n_harmonics = 5
        season_features = []
        for k in range(1, n_harmonics + 1):
            season_features.append(np.sin(2 * np.pi * k * week_of_year / 52))
            season_features.append(np.cos(2 * np.pi * k * week_of_year / 52))
        season_matrix = np.column_stack(season_features)

        # Get priors
        def col_to_prior_name(col):
            if col == "email_opens":
                return "email"
            return col.replace("_spend", "")

        channel_priors = [get_channel_prior(col_to_prior_name(c)) for c in spend_cols]

        # Channel params per channel:
        #   [decay, alpha, lam, lam_trend, beta_base, beta_trend, beta_event]  = 7 if interactions
        #   [decay, alpha, lam, lam_trend, beta_base, beta_trend]              = 6 if no interactions
        # beta_trend: vertical shift (same spend → more revenue)
        # lam_trend: horizontal shift (saturation curve expands → can spend more)
        params_per_channel = 7 if has_event_interactions else 6

        # Cross-channel spillover (e.g., Meta → Google brand search halo)
        channel_names = [col_to_prior_name(c) for c in spend_cols]
        spillover_pairs_raw = get_spillover_pairs(channel_names)
        # Convert channel names to indices
        spillover_pairs = []
        for src_name, tgt_name, prior_mean, prior_sd in spillover_pairs_raw:
            src_idx = channel_names.index(src_name)
            tgt_idx = channel_names.index(tgt_name)
            spillover_pairs.append((src_idx, tgt_idx, prior_mean, prior_sd))
            logger.info(f"Spillover enabled: {src_name} → {tgt_name} "
                       f"(prior: {prior_mean:.2f} ± {prior_sd:.2f})")
        n_spillover = len(spillover_pairs)

        def model_predict_norm(params):
            """Predict revenue in normalized (z-score) space.

            Two-pass approach to support cross-channel spillover:
            1. Compute all adstocked signals
            2. Apply spillover (e.g., Meta's adstock flows into Google's signal)
            3. Compute saturation and channel effects
            """
            idx = 0
            pred = np.full(T, params[idx])  # intercept (≈0 in z-space)
            idx += 1

            # Linear trend
            pred += params[idx] * time_index
            idx += 1

            # Seasonality
            for j in range(season_matrix.shape[1]):
                pred += params[idx] * season_matrix[:, j]
                idx += 1

            ch_start = idx  # remember where channel params begin

            # Pass 1: compute adstocked signals for all channels
            adstocked_signals = []
            for i in range(n_channels):
                base = ch_start + i * params_per_channel
                decay = _safe_sigmoid(params[base])
                adstocked = geometric_adstock(spend_matrix[:, i], decay, channel_priors[i].adstock_max_lag)
                adstocked_signals.append(adstocked)

            # Apply spillover: add source's adstocked signal to target
            # Spillover params are at the end of the parameter vector
            spill_start = ch_start + n_channels * params_per_channel + n_controls
            for s, (src, tgt, _, _) in enumerate(spillover_pairs):
                frac = _safe_sigmoid(params[spill_start + s])
                adstocked_signals[tgt] = adstocked_signals[tgt] + frac * adstocked_signals[src]

            # Pass 2: compute saturation and add channel effects to prediction
            for i in range(n_channels):
                base = ch_start + i * params_per_channel
                alpha = _safe_exp(params[base + 1])
                lam_base = _safe_exp(params[base + 2])
                lam_trend = params[base + 3]
                beta_base = _safe_exp(params[base + 4])
                beta_trend = params[base + 5]

                effective_lam = lam_base * _safe_exp(lam_trend * time_index)
                effective_beta = beta_base + beta_trend * time_index

                if has_event_interactions:
                    beta_event = _safe_exp(params[base + 6])
                    effective_beta = effective_beta + beta_event * heavy_discount

                adstocked = adstocked_signals[i]
                x_norm = adstocked / (adstocked.mean() + 1e-8)
                alpha_c = np.clip(alpha, 0.01, 10.0)
                exponent = np.clip(effective_lam * np.power(x_norm, alpha_c), 0, 30)
                saturated = 1 - np.exp(-exponent)
                pred += effective_beta * saturated

            # Control effects (in z-score units)
            ctrl_start = ch_start + n_channels * params_per_channel
            for j in range(control_matrix.shape[1]):
                pred += params[ctrl_start + j] * control_matrix[:, j]

            return pred

        def model_predict(params):
            """Predict revenue in original (raw) scale."""
            return model_predict_norm(params) * y_std + y_mean

        def loss(params):
            """Negative log posterior = MSE + prior regularization."""
            pred_norm = model_predict_norm(params)

            # Data likelihood (MSE in normalized space)
            mse = np.mean((y_norm - pred_norm) ** 2)

            # Prior regularization on channel params
            reg = 0.0
            idx = 1 + 1 + season_matrix.shape[1]  # skip intercept + trend + season
            for i in range(n_channels):
                prior = channel_priors[i]
                decay = _safe_sigmoid(params[idx])
                reg += 0.5 * ((decay - prior.adstock_decay_mean) / prior.adstock_decay_sd) ** 2

                alpha = _safe_exp(params[idx + 1])
                reg += 0.5 * ((alpha - prior.saturation_alpha_mean) / prior.saturation_alpha_sd) ** 2

                lam_base = _safe_exp(params[idx + 2])
                reg += 0.5 * ((lam_base - prior.saturation_lam_mean) / prior.saturation_lam_sd) ** 2

                # lam_trend: light L2 centered at 0 (no capacity change by default)
                # SD=0.5 means ±50% change in lam over the training period is ~1σ
                lam_trend = params[idx + 3]
                reg += 0.5 * (lam_trend / 0.5) ** 2

                beta_base = _safe_exp(params[idx + 4])
                reg += 0.5 * ((beta_base - prior.beta_mean) / prior.beta_sd) ** 2

                # beta_trend: light L2 centered at 0 (no trend by default)
                beta_trend = params[idx + 5]
                reg += 0.5 * (beta_trend / 0.5) ** 2

                if has_event_interactions:
                    # Very light regularization on event boost — we expect it to
                    # potentially be much larger than base beta (3-5x during
                    # Black Week). Use 5x prior SD to let the data speak.
                    beta_event = _safe_exp(params[idx + 6])
                    reg += 0.5 * ((beta_event - prior.beta_mean) / (prior.beta_sd * 5)) ** 2

                idx += params_per_channel

            # Spillover prior regularization
            spill_start = 1 + 1 + season_matrix.shape[1] + n_channels * params_per_channel + n_controls
            for s, (_, _, prior_mean, prior_sd) in enumerate(spillover_pairs):
                frac = _safe_sigmoid(params[spill_start + s])
                reg += 0.5 * ((frac - prior_mean) / prior_sd) ** 2

            # Return inf for NaN to guide optimizer away
            total = mse + 0.005 * reg
            return total if np.isfinite(total) else 1e10

        # Initialize parameters (all in z-score space)
        n_season = season_matrix.shape[1]
        n_controls = control_matrix.shape[1]
        n_params = 1 + 1 + n_season + n_channels * params_per_channel + n_controls + n_spillover

        x0 = np.zeros(n_params)
        idx = 1 + 1 + n_season  # skip intercept + trend + season
        for i, prior in enumerate(channel_priors):
            x0[idx] = np.log(prior.adstock_decay_mean / (1 - prior.adstock_decay_mean + 1e-8))
            x0[idx + 1] = np.log(prior.saturation_alpha_mean)
            x0[idx + 2] = np.log(prior.saturation_lam_mean)
            x0[idx + 3] = 0.0  # lam_trend: no capacity change initially
            x0[idx + 4] = np.log(prior.beta_mean)
            x0[idx + 5] = 0.0  # beta_trend: no trend initially
            if has_event_interactions:
                x0[idx + 6] = np.log(prior.beta_mean)  # init event boost = base beta
            idx += params_per_channel

        # Initialize spillover params (in logit space)
        spill_start = 1 + 1 + n_season + n_channels * params_per_channel + n_controls
        for s, (_, _, prior_mean, _) in enumerate(spillover_pairs):
            # logit(prior_mean) as initial value
            x0[spill_start + s] = np.log(prior_mean / (1 - prior_mean + 1e-8))

        # Multi-start optimization for robustness
        logger.info(f"Fitting MMM with {n_channels} channels, {T} weeks...")
        best_result = None
        best_loss = np.inf

        for start_idx in range(3):
            if start_idx == 0:
                x_init = x0.copy()
            else:
                # Perturbed start
                x_init = x0 + np.random.randn(n_params) * 0.3
            try:
                res = minimize(loss, x_init, method="L-BFGS-B",
                             options={"maxiter": 2000, "disp": False})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = res
            except Exception as e:
                logger.warning(f"Optimization start {start_idx} failed: {e}")

        if best_result is None or not best_result.success:
            logger.warning(f"Optimization did not fully converge: "
                          f"{best_result.message if best_result else 'all starts failed'}")

        best_params = best_result.x

        # Bootstrap for uncertainty quantification
        n_bootstrap = 25
        bootstrap_params = []
        logger.info(f"Running {n_bootstrap} bootstrap iterations...")
        for b in range(n_bootstrap):
            idx_boot = np.random.choice(T, size=T, replace=True)
            y_boot = y[idx_boot]
            y_boot_mean = y_boot.mean()
            y_boot_std = y_boot.std() + 1e-8
            y_boot_norm = (y_boot - y_boot_mean) / y_boot_std
            spend_boot = spend_matrix[idx_boot]
            control_boot = control_matrix[idx_boot]
            season_boot = season_matrix[idx_boot]
            time_boot = time_index[idx_boot]
            heavy_boot = heavy_discount[idx_boot]

            def loss_boot(params, _y_norm=y_boot_norm, _spend=spend_boot,
                         _control=control_boot, _season=season_boot, _time=time_boot,
                         _heavy=heavy_boot):
                idx = 0
                pred = np.full(T, params[idx])
                idx += 1
                pred += params[idx] * _time
                idx += 1
                for j in range(_season.shape[1]):
                    pred += params[idx] * _season[:, j]
                    idx += 1
                ch_start = idx
                # Pass 1: adstock all channels
                b_adstocked = []
                for i in range(n_channels):
                    base = ch_start + i * params_per_channel
                    decay = _safe_sigmoid(params[base])
                    adstocked = geometric_adstock(_spend[:, i], decay, channel_priors[i].adstock_max_lag)
                    b_adstocked.append(adstocked)
                # Apply spillover
                b_spill_start = ch_start + n_channels * params_per_channel + _control.shape[1]
                for s, (src, tgt, _, _) in enumerate(spillover_pairs):
                    frac = _safe_sigmoid(params[b_spill_start + s])
                    b_adstocked[tgt] = b_adstocked[tgt] + frac * b_adstocked[src]
                # Pass 2: saturation + contribution
                for i in range(n_channels):
                    base = ch_start + i * params_per_channel
                    alpha = _safe_exp(params[base + 1])
                    lam_base = _safe_exp(params[base + 2])
                    lam_trend = params[base + 3]
                    beta_base = _safe_exp(params[base + 4])
                    beta_trend = params[base + 5]
                    effective_lam = lam_base * _safe_exp(lam_trend * _time)
                    effective_beta = beta_base + beta_trend * _time
                    if has_event_interactions:
                        beta_event = _safe_exp(params[base + 6])
                        effective_beta = effective_beta + beta_event * _heavy
                    adstocked = b_adstocked[i]
                    x_norm = adstocked / (adstocked.mean() + 1e-8)
                    alpha_c = np.clip(alpha, 0.01, 10.0)
                    exponent = np.clip(effective_lam * np.power(x_norm, alpha_c), 0, 30)
                    saturated = 1 - np.exp(-exponent)
                    pred += effective_beta * saturated
                ctrl_start = ch_start + n_channels * params_per_channel
                for j in range(_control.shape[1]):
                    pred += params[ctrl_start + j] * _control[:, j]
                val = np.mean((_y_norm - pred) ** 2)
                return val if np.isfinite(val) else 1e10

            try:
                res_boot = minimize(loss_boot, best_params, method="L-BFGS-B",
                                  options={"maxiter": 200, "disp": False})
                bootstrap_params.append(res_boot.x)
            except Exception:
                bootstrap_params.append(best_params)

        bootstrap_params = np.array(bootstrap_params)

        # ── Extract results ──────────────────────────────────────

        # Full prediction (raw scale)
        y_pred = model_predict(best_params)

        # Decompose into components (raw scale)
        # Intercept + trend + seasonality = baseline
        intercept_norm = best_params[0]
        trend_norm = best_params[1] * time_index
        season_norm = np.zeros(T)
        for j in range(n_season):
            season_norm += best_params[2 + j] * season_matrix[:, j]

        # Convert from z-score to raw scale
        baseline_raw = (intercept_norm + trend_norm + season_norm) * y_std + y_mean
        seasonality_raw = season_norm * y_std
        trend_raw = trend_norm * y_std

        # Channel contributions (raw scale)
        # Two-pass: first adstock all channels, then apply spillover, then saturate
        contributions = {}
        channel_params_dict = {}
        ch_start = 1 + 1 + n_season

        # Pass 1: compute raw adstocked signals for all channels
        raw_adstocked = []
        for i in range(n_channels):
            base = ch_start + i * params_per_channel
            decay = _safe_sigmoid(best_params[base])
            adstocked = geometric_adstock(spend_matrix[:, i], decay, channel_priors[i].adstock_max_lag)
            raw_adstocked.append(adstocked)

        # Extract spillover fractions
        spill_param_start = ch_start + n_channels * params_per_channel + n_controls
        spillover_fracs = {}
        for s, (src, tgt, _, _) in enumerate(spillover_pairs):
            frac = float(_safe_sigmoid(best_params[spill_param_start + s]))
            spillover_fracs[(src, tgt)] = frac
            src_name = channel_names[src]
            tgt_name = channel_names[tgt]
            logger.info(f"Spillover {src_name} → {tgt_name}: {frac:.1%}")

        # Apply spillover to get effective adstocked signals
        effective_adstocked = [a.copy() for a in raw_adstocked]
        for (src, tgt), frac in spillover_fracs.items():
            effective_adstocked[tgt] = effective_adstocked[tgt] + frac * raw_adstocked[src]

        # Pass 2: compute saturation and contributions
        # For spillover targets, also compute counterfactual (without spillover)
        # to separate the spillover contribution for reattribution
        spillover_deltas = {}  # (src, tgt) -> delta contribution array
        for i, col in enumerate(spend_cols):
            channel_name = col_to_prior_name(col)
            base = ch_start + i * params_per_channel
            decay = _safe_sigmoid(best_params[base])
            alpha = _safe_exp(best_params[base + 1])
            lam_base = _safe_exp(best_params[base + 2])
            lam_trend = best_params[base + 3]
            beta_base = _safe_exp(best_params[base + 4])
            beta_trend = best_params[base + 5]

            effective_lam = lam_base * _safe_exp(lam_trend * time_index)
            effective_beta = beta_base + beta_trend * time_index

            if has_event_interactions:
                beta_event = _safe_exp(best_params[base + 6])
                effective_beta = effective_beta + beta_event * heavy_discount
            else:
                beta_event = 0.0

            # Contribution with spillover (what model actually predicts)
            adstocked = effective_adstocked[i]
            x_norm = adstocked / (adstocked.mean() + 1e-8)
            alpha_c = np.clip(alpha, 0.01, 10.0)
            exponent = np.clip(effective_lam * np.power(x_norm, alpha_c), 0, 30)
            saturated = 1 - np.exp(-exponent)
            contribution_raw = effective_beta * saturated * y_std

            # Check if this channel is a spillover target
            for (src, tgt), frac in spillover_fracs.items():
                if tgt == i:
                    # Counterfactual: what would contribution be without spillover?
                    own_adstocked = raw_adstocked[i]
                    own_x_norm = own_adstocked / (own_adstocked.mean() + 1e-8)
                    own_exponent = np.clip(effective_lam * np.power(own_x_norm, alpha_c), 0, 30)
                    own_saturated = 1 - np.exp(-own_exponent)
                    own_contribution = effective_beta * own_saturated * y_std
                    spillover_deltas[(src, tgt)] = contribution_raw - own_contribution
                    logger.info(f"Spillover reattribution: {channel_names[src]} → {channel_names[tgt]}: "
                               f"{(contribution_raw - own_contribution).sum():,.0f} revenue reattributed")

            # Values at end of training period (most recent)
            beta_end = beta_base + beta_trend * 1.0
            lam_end = float(lam_base * np.exp(lam_trend * 1.0))
            beta_trend_pct = beta_trend / (beta_base + 1e-8) * 100
            lam_change_pct = (np.exp(lam_trend) - 1) * 100

            contributions[channel_name] = contribution_raw
            channel_params_dict[channel_name] = {
                "adstock_decay": float(decay),
                "saturation_alpha": float(alpha),
                "saturation_lam": float(lam_base),
                "saturation_lam_trend": float(lam_trend),
                "saturation_lam_end": float(lam_end),
                "saturation_lam_change_pct": float(lam_change_pct),
                "beta": float(beta_base),
                "beta_raw": float(beta_base * y_std),
                "beta_trend": float(beta_trend),
                "beta_trend_pct": float(beta_trend_pct),
                "beta_end": float(beta_end),
                "beta_end_raw": float(beta_end * y_std),
                "beta_event": float(beta_event),
                "beta_event_raw": float(beta_event * y_std),
                "event_multiplier": float(1 + beta_event / (beta_end + 1e-8)),
                "adstock_training_mean": float(effective_adstocked[i].mean()),
            }

        # Reattribute spillover: move delta from target to source
        for (src, tgt), delta in spillover_deltas.items():
            src_name = channel_names[src]
            tgt_name = channel_names[tgt]
            contributions[src_name] = contributions[src_name] + delta
            contributions[tgt_name] = contributions[tgt_name] - delta
            # Record spillover info in channel params
            channel_params_dict[src_name]["spillover_to"] = tgt_name
            channel_params_dict[src_name]["spillover_fraction"] = spillover_fracs[(src, tgt)]
            channel_params_dict[src_name]["spillover_contribution"] = float(delta.sum())
            channel_params_dict[tgt_name]["spillover_from"] = src_name
            channel_params_dict[tgt_name]["spillover_reduction"] = float(delta.sum())

        # Control contributions (raw scale)
        control_contribs = {}
        ctrl_start = ch_start + n_channels * params_per_channel
        for j, ctrl_col in enumerate(control_cols):
            control_contribs[ctrl_col] = best_params[ctrl_start + j] * control_matrix[:, j] * y_std

        # Channel ROAS with bootstrap CI (uses reattributed contributions)
        roas_data = []
        for i, col in enumerate(spend_cols):
            channel_name = col_to_prior_name(col)
            total_spend = spend_matrix[:, i].sum()
            total_contribution = contributions[channel_name].sum()
            roas_mean = total_contribution / (total_spend + 1e-8)

            # Bootstrap ROAS with spillover-aware decomposition
            boot_roas = []
            for b_params in bootstrap_params:
                # Adstock all channels
                b_raw_adstocked = []
                for j in range(n_channels):
                    b_base = ch_start + j * params_per_channel
                    b_decay = _safe_sigmoid(b_params[b_base])
                    b_adstocked = geometric_adstock(spend_matrix[:, j], b_decay, channel_priors[j].adstock_max_lag)
                    b_raw_adstocked.append(b_adstocked)

                # Apply spillover
                b_eff_adstocked = [a.copy() for a in b_raw_adstocked]
                for s, (src, tgt, _, _) in enumerate(spillover_pairs):
                    b_frac = _safe_sigmoid(b_params[spill_param_start + s])
                    b_eff_adstocked[tgt] = b_eff_adstocked[tgt] + b_frac * b_raw_adstocked[src]

                # Compute contribution for channel i
                b_base_i = ch_start + i * params_per_channel
                b_alpha = _safe_exp(b_params[b_base_i + 1])
                b_lam_base = _safe_exp(b_params[b_base_i + 2])
                b_lam_trend = b_params[b_base_i + 3]
                b_beta_base = _safe_exp(b_params[b_base_i + 4])
                b_beta_trend = b_params[b_base_i + 5]
                b_effective_lam = b_lam_base * _safe_exp(b_lam_trend * time_index)
                b_effective_beta = b_beta_base + b_beta_trend * time_index
                if has_event_interactions:
                    b_beta_event = _safe_exp(b_params[b_base_i + 6])
                    b_effective_beta = b_effective_beta + b_beta_event * heavy_discount

                b_adstocked = b_eff_adstocked[i]
                b_x_norm = b_adstocked / (b_adstocked.mean() + 1e-8)
                b_alpha_c = np.clip(b_alpha, 0.01, 10.0)
                b_exponent = np.clip(b_effective_lam * np.power(b_x_norm, b_alpha_c), 0, 30)
                b_saturated = 1 - np.exp(-b_exponent)
                b_contribution = (b_effective_beta * b_saturated).sum() * y_std

                # Apply spillover reattribution for this channel
                for (src, tgt), _ in spillover_fracs.items():
                    if tgt == i:
                        # Subtract spillover delta
                        b_own = b_raw_adstocked[i]
                        b_own_xn = b_own / (b_own.mean() + 1e-8)
                        b_own_exp = np.clip(b_effective_lam * np.power(b_own_xn, b_alpha_c), 0, 30)
                        b_own_sat = 1 - np.exp(-b_own_exp)
                        b_own_contrib = (b_effective_beta * b_own_sat).sum() * y_std
                        b_contribution = b_own_contrib  # use own contribution only
                    elif src == i:
                        # Add spillover delta
                        tgt_base = ch_start + tgt * params_per_channel
                        bt_alpha = _safe_exp(b_params[tgt_base + 1])
                        bt_lam_base = _safe_exp(b_params[tgt_base + 2])
                        bt_lam_trend = b_params[tgt_base + 3]
                        bt_beta_base = _safe_exp(b_params[tgt_base + 4])
                        bt_beta_trend = b_params[tgt_base + 5]
                        bt_eff_lam = bt_lam_base * _safe_exp(bt_lam_trend * time_index)
                        bt_eff_beta = bt_beta_base + bt_beta_trend * time_index
                        if has_event_interactions:
                            bt_beta_ev = _safe_exp(b_params[tgt_base + 6])
                            bt_eff_beta = bt_eff_beta + bt_beta_ev * heavy_discount
                        bt_alpha_c = np.clip(bt_alpha, 0.01, 10.0)
                        # With spillover
                        bt_ad_with = b_eff_adstocked[tgt]
                        bt_xn_w = bt_ad_with / (bt_ad_with.mean() + 1e-8)
                        bt_exp_w = np.clip(bt_eff_lam * np.power(bt_xn_w, bt_alpha_c), 0, 30)
                        bt_contrib_w = (bt_eff_beta * (1 - np.exp(-bt_exp_w))).sum() * y_std
                        # Without spillover
                        bt_ad_own = b_raw_adstocked[tgt]
                        bt_xn_o = bt_ad_own / (bt_ad_own.mean() + 1e-8)
                        bt_exp_o = np.clip(bt_eff_lam * np.power(bt_xn_o, bt_alpha_c), 0, 30)
                        bt_contrib_o = (bt_eff_beta * (1 - np.exp(-bt_exp_o))).sum() * y_std
                        b_contribution += (bt_contrib_w - bt_contrib_o)

                boot_roas.append(b_contribution / (total_spend + 1e-8))

            roas_data.append({
                "channel": channel_name,
                "total_spend": total_spend,
                "total_contribution": total_contribution,
                "roas_mean": roas_mean,
                "roas_5": np.percentile(boot_roas, 5),
                "roas_25": np.percentile(boot_roas, 25),
                "roas_50": np.percentile(boot_roas, 50),
                "roas_75": np.percentile(boot_roas, 75),
                "roas_95": np.percentile(boot_roas, 95),
            })

        # Build DataFrames
        contrib_df = pd.DataFrame({"week_start": df["week_start"]})
        for name, vals in contributions.items():
            contrib_df[name] = vals

        control_df = pd.DataFrame({"week_start": df["week_start"]})
        for name, vals in control_contribs.items():
            control_df[name] = vals

        roas_df = pd.DataFrame(roas_data)

        # Model fit metrics
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot
        mape = np.mean(np.abs(residuals) / (y + 1e-8)) * 100

        results = MMMResults(
            channel_contributions=contrib_df,
            channel_roas=roas_df,
            channel_params=channel_params_dict,
            baseline_contribution=baseline_raw,
            control_contributions=control_df,
            seasonality=seasonality_raw,
            actual=y,
            predicted=y_pred,
            residuals=residuals,
            r_squared=r_squared,
            mape=mape,
            posterior_samples={"bootstrap_params": bootstrap_params},
            spend_columns=spend_cols,
            date_range=(str(df["week_start"].min().date()), str(df["week_start"].max().date())),
            n_weeks=T,
            target_col=target_col,
        )

        logger.info(f"Model fit complete: R²={r_squared:.3f}, MAPE={mape:.1f}%")
        return results


# ═══════════════════════════════════════════════════════════════
# MERIDIAN WRAPPER (for when Meridian is installed)
# ═══════════════════════════════════════════════════════════════

class MeridianMMM:
    """
    Wrapper around Google's Meridian framework.

    Install with: pip install google-meridian

    Falls back to LightweightMMM if Meridian is not installed.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        try:
            import meridian
            self._backend = "meridian"
            logger.info("Using Google Meridian backend")
        except ImportError:
            logger.info("Meridian not installed. Using lightweight built-in MMM. "
                       "Install with: pip install google-meridian")
            self._backend = "lightweight"
            self._fallback = LightweightMMM(config)

    def fit(self, df, target_col="revenue", spend_cols=None, control_cols=None):
        if self._backend == "lightweight":
            return self._fallback.fit(df, target_col, spend_cols, control_cols)

        # Meridian-specific implementation
        # (This would use meridian.InputData, meridian.Meridian, etc.)
        # For now, use the lightweight version
        logger.info("Meridian backend: using lightweight implementation")
        return LightweightMMM(self.config).fit(df, target_col, spend_cols, control_cols)


def create_model(config: Optional[dict] = None) -> LightweightMMM:
    """Factory function: create the best available MMM backend."""
    try:
        return MeridianMMM(config)
    except Exception:
        return LightweightMMM(config)
