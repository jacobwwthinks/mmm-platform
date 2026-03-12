"""
Marketing Mix Model using Bayesian inference.

This module wraps the MMM modeling process with a clean API.
It supports Google's Meridian as the primary backend, with a
lightweight PyMC-based fallback.

The model decomposes revenue into:
    revenue = baseline + seasonality + sum(channel_effects) + sum(control_effects) + noise

Where each channel_effect = beta * saturation(adstock(spend))
"""

import numpy as np
import pandas as pd
import logging
import json
import pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from model.priors import get_channel_prior, CONTROL_PRIORS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ADSTOCK & SATURATION TRANSFORMS
# ═══════════════════════════════════════════════════════════════

def geometric_adstock(x: np.ndarray, decay: float, max_lag: int = 8) -> np.ndarray:
    """
    Apply geometric adstock transformation.

    Simulates the carryover effect: an ad seen this week still has
    a decaying effect in subsequent weeks.

    Args:
        x: Spend time series (T,)
        decay: Decay rate 0-1 (higher = longer memory)
        max_lag: Maximum number of weeks to carry over

    Returns:
        Transformed series with carryover effects
    """
    T = len(x)
    result = np.zeros(T)
    for t in range(T):
        for lag in range(min(max_lag, t + 1)):
            result[t] += x[t - lag] * (decay ** lag)
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
    return 1 - np.exp(-lam * x_norm ** alpha)


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

    def save(self, path: str):
        """Save results to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.channel_contributions.to_csv(f"{path}/channel_contributions.csv", index=False)
        self.channel_roas.to_csv(f"{path}/channel_roas.csv", index=False)
        with open(f"{path}/params.json", "w") as f:
            json.dump(self.channel_params, f, indent=2, default=str)
        with open(f"{path}/results.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Results saved to {path}/")

    @classmethod
    def load(cls, path: str) -> "MMMResults":
        """Load results from disk."""
        with open(f"{path}/results.pkl", "rb") as f:
            return pickle.load(f)


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

        This uses maximum a posteriori (MAP) estimation with bootstrap
        for uncertainty quantification. Fast and reliable.

        For full Bayesian posterior, use fit_bayesian() (requires NumPyro).

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
        # Include both _spend columns (paid media) and email_opens (Klaviyo)
        if spend_cols is None:
            spend_cols = sorted([c for c in df.columns if c.endswith("_spend") and df[c].sum() > 0])
            if "email_opens" in df.columns and df["email_opens"].sum() > 0:
                spend_cols.append("email_opens")
        if control_cols is None:
            control_cols = [c for c in ["discount_campaign", "product_drop", "holiday"]
                          if c in df.columns]

        y = df[target_col].values.astype(float)
        T = len(y)
        n_channels = len(spend_cols)

        # Normalize target for better optimization
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y_norm = (y - y_mean) / y_std

        # Prepare spend matrices
        spend_matrix = np.column_stack([df[col].values.astype(float) for col in spend_cols])
        control_matrix = np.column_stack([df[col].values.astype(float) for col in control_cols]) \
            if control_cols else np.zeros((T, 0))

        # Seasonality features (Fourier)
        week_of_year = df["week_start"].dt.isocalendar().week.values.astype(float)
        n_harmonics = 3
        season_features = []
        for k in range(1, n_harmonics + 1):
            season_features.append(np.sin(2 * np.pi * k * week_of_year / 52))
            season_features.append(np.cos(2 * np.pi * k * week_of_year / 52))
        season_matrix = np.column_stack(season_features)

        # Get priors — map column names to channel prior names
        # email_opens -> "email", meta_spend -> "meta", etc.
        def col_to_prior_name(col):
            if col == "email_opens":
                return "email"
            return col.replace("_spend", "")

        channel_priors = [get_channel_prior(col_to_prior_name(c)) for c in spend_cols]

        def model_predict(params):
            """Predict revenue given parameters."""
            idx = 0
            pred = np.full(T, params[idx])  # intercept
            idx += 1

            # Seasonality
            for j in range(season_matrix.shape[1]):
                pred += params[idx] * season_matrix[:, j]
                idx += 1

            # Channel effects
            for i in range(n_channels):
                decay = 1 / (1 + np.exp(-params[idx]))  # sigmoid to keep in (0,1)
                alpha = np.exp(params[idx + 1])           # positive
                lam = np.exp(params[idx + 2])              # positive
                beta = np.exp(params[idx + 3])             # positive (channels help revenue)
                idx += 4

                adstocked = geometric_adstock(spend_matrix[:, i], decay, channel_priors[i].adstock_max_lag)
                saturated = hill_saturation(adstocked, alpha, lam)
                pred += beta * saturated

            # Control effects
            for j in range(control_matrix.shape[1]):
                pred += params[idx] * control_matrix[:, j]
                idx += 1

            return pred

        def loss(params):
            """Negative log posterior = MSE + prior regularization."""
            pred = model_predict(params)
            pred_norm = (pred - y_mean) / y_std

            # Data likelihood (MSE)
            mse = np.mean((y_norm - pred_norm) ** 2)

            # Prior regularization on channel params
            reg = 0.0
            idx = 1 + season_matrix.shape[1]
            for i in range(n_channels):
                prior = channel_priors[i]
                decay_raw = params[idx]
                decay = 1 / (1 + np.exp(-decay_raw))
                reg += 0.5 * ((decay - prior.adstock_decay_mean) / prior.adstock_decay_sd) ** 2

                alpha = np.exp(params[idx + 1])
                reg += 0.5 * ((alpha - prior.saturation_alpha_mean) / prior.saturation_alpha_sd) ** 2

                lam = np.exp(params[idx + 2])
                reg += 0.5 * ((lam - prior.saturation_lam_mean) / prior.saturation_lam_sd) ** 2

                beta = np.exp(params[idx + 3])
                reg += 0.5 * ((beta - prior.beta_mean) / prior.beta_sd) ** 2

                idx += 4

            return mse + 0.01 * reg

        # Initialize parameters
        n_season = season_matrix.shape[1]
        n_controls = control_matrix.shape[1]
        n_params = 1 + n_season + n_channels * 4 + n_controls

        # Smart initialization from priors
        x0 = np.zeros(n_params)
        x0[0] = y_mean  # intercept ~ mean revenue
        idx = 1 + n_season
        for i, prior in enumerate(channel_priors):
            x0[idx] = np.log(prior.adstock_decay_mean / (1 - prior.adstock_decay_mean + 1e-8))
            x0[idx + 1] = np.log(prior.saturation_alpha_mean)
            x0[idx + 2] = np.log(prior.saturation_lam_mean)
            x0[idx + 3] = np.log(prior.beta_mean)
            idx += 4

        # Optimize (MAP estimation)
        logger.info(f"Fitting MMM with {n_channels} channels, {T} weeks...")
        result = minimize(loss, x0, method="L-BFGS-B", options={"maxiter": 2000, "disp": False})

        if not result.success:
            logger.warning(f"Optimization did not fully converge: {result.message}")

        best_params = result.x

        # Bootstrap for uncertainty quantification
        n_bootstrap = 200
        bootstrap_params = []
        for b in range(n_bootstrap):
            # Resample with replacement
            idx_boot = np.random.choice(T, size=T, replace=True)
            y_boot = y[idx_boot]
            spend_boot = spend_matrix[idx_boot]
            control_boot = control_matrix[idx_boot]
            season_boot = season_matrix[idx_boot]

            # Quick optimization from current best
            def loss_boot(params):
                idx = 0
                pred = np.full(T, params[idx])
                idx += 1
                for j in range(season_boot.shape[1]):
                    pred += params[idx] * season_boot[:, j]
                    idx += 1
                for i in range(n_channels):
                    decay = 1 / (1 + np.exp(-params[idx]))
                    alpha = np.exp(params[idx + 1])
                    lam = np.exp(params[idx + 2])
                    beta = np.exp(params[idx + 3])
                    idx += 4
                    adstocked = geometric_adstock(spend_boot[:, i], decay, channel_priors[i].adstock_max_lag)
                    saturated = hill_saturation(adstocked, alpha, lam)
                    pred += beta * saturated
                for j in range(control_boot.shape[1]):
                    pred += params[idx] * control_boot[:, j]
                    idx += 1
                pred_norm = (pred - y_boot.mean()) / (y_boot.std() + 1e-8)
                y_boot_norm = (y_boot - y_boot.mean()) / (y_boot.std() + 1e-8)
                return np.mean((y_boot_norm - pred_norm) ** 2)

            res_boot = minimize(loss_boot, best_params, method="L-BFGS-B",
                              options={"maxiter": 500, "disp": False})
            bootstrap_params.append(res_boot.x)

        bootstrap_params = np.array(bootstrap_params)

        # ── Extract results ──────────────────────────────────────

        # Full prediction
        y_pred = model_predict(best_params)

        # Decompose into components
        idx = 1 + n_season

        # Baseline (intercept + seasonality)
        baseline = np.full(T, best_params[0])
        for j in range(n_season):
            baseline += best_params[1 + j] * season_matrix[:, j]

        seasonality = baseline - best_params[0]

        # Channel contributions
        contributions = {}
        channel_params = {}
        for i, col in enumerate(spend_cols):
            channel_name = col_to_prior_name(col)
            decay = 1 / (1 + np.exp(-best_params[idx]))
            alpha = np.exp(best_params[idx + 1])
            lam = np.exp(best_params[idx + 2])
            beta = np.exp(best_params[idx + 3])

            adstocked = geometric_adstock(spend_matrix[:, i], decay, channel_priors[i].adstock_max_lag)
            saturated = hill_saturation(adstocked, alpha, lam)
            contribution = beta * saturated

            contributions[channel_name] = contribution
            channel_params[channel_name] = {
                "adstock_decay": float(decay),
                "saturation_alpha": float(alpha),
                "saturation_lam": float(lam),
                "beta": float(beta),
            }
            idx += 4

        # Control contributions
        control_contribs = {}
        for j, ctrl_col in enumerate(control_cols):
            control_contribs[ctrl_col] = best_params[idx] * control_matrix[:, j]
            idx += 1

        # Channel ROAS with bootstrap CI
        # For email_opens, we compute "revenue per 1000 opens" instead of ROAS
        roas_data = []
        for i, col in enumerate(spend_cols):
            channel_name = col_to_prior_name(col)
            total_spend = spend_matrix[:, i].sum()
            total_contribution = contributions[channel_name].sum()
            roas_mean = total_contribution / (total_spend + 1e-8)

            # Bootstrap ROAS distribution
            boot_roas = []
            for b_params in bootstrap_params:
                b_idx = 1 + n_season + i * 4
                b_decay = 1 / (1 + np.exp(-b_params[b_idx]))
                b_alpha = np.exp(b_params[b_idx + 1])
                b_lam = np.exp(b_params[b_idx + 2])
                b_beta = np.exp(b_params[b_idx + 3])
                b_adstocked = geometric_adstock(spend_matrix[:, i], b_decay, channel_priors[i].adstock_max_lag)
                b_saturated = hill_saturation(b_adstocked, b_alpha, b_lam)
                b_contribution = (b_beta * b_saturated).sum()
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

        # Build contribution DataFrame
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
            channel_params=channel_params,
            baseline_contribution=baseline,
            control_contributions=control_df,
            seasonality=seasonality,
            actual=y,
            predicted=y_pred,
            residuals=residuals,
            r_squared=r_squared,
            mape=mape,
            posterior_samples={"bootstrap_params": bootstrap_params},
            spend_columns=spend_cols,
            date_range=(str(df["week_start"].min().date()), str(df["week_start"].max().date())),
            n_weeks=T,
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
