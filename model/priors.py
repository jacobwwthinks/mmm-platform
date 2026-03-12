"""
DTC e-commerce channel priors for Bayesian MMM.

These priors encode domain knowledge about how different marketing channels
typically perform in DTC/e-commerce. They serve as starting beliefs that
the model updates with actual data.

Based on industry benchmarks and Meta/Google MMM research papers.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChannelPrior:
    """Prior beliefs about a marketing channel's behavior."""

    # --- Adstock (carryover effect) ---
    # How long does the effect of an ad persist after it's shown?
    adstock_decay_mean: float     # Mean decay rate (0-1). Higher = longer memory
    adstock_decay_sd: float       # Uncertainty on decay rate
    adstock_max_lag: int          # Maximum weeks to look back

    # --- Saturation (diminishing returns) ---
    # At what point does more spend stop producing proportional returns?
    saturation_alpha_mean: float  # Shape parameter: steepness of saturation
    saturation_alpha_sd: float
    saturation_lam_mean: float    # Scale parameter: where saturation kicks in
    saturation_lam_sd: float

    # --- Channel coefficient (effect size) ---
    beta_mean: float              # Expected coefficient (effect on revenue)
    beta_sd: float                # Uncertainty

    # --- Description ---
    notes: str = ""


# ═══════════════════════════════════════════════════════════════
# DTC E-COMMERCE CHANNEL PRIORS
# ═══════════════════════════════════════════════════════════════

DTC_CHANNEL_PRIORS = {

    "meta": ChannelPrior(
        # Meta (Facebook/Instagram): Broad reach, moderate carryover.
        # Brand awareness ads have longer decay; conversion ads shorter.
        adstock_decay_mean=0.4,
        adstock_decay_sd=0.15,
        adstock_max_lag=4,
        saturation_alpha_mean=2.0,
        saturation_alpha_sd=1.0,
        saturation_lam_mean=0.5,
        saturation_lam_sd=0.3,
        beta_mean=0.15,
        beta_sd=0.1,
        notes="Primary acquisition channel for most DTC brands. "
              "Mix of prospecting + retargeting. Moderate carryover."
    ),

    "google_ads": ChannelPrior(
        # Google Ads (primarily Search): High intent, short carryover.
        # People search when they're ready to buy.
        adstock_decay_mean=0.2,
        adstock_decay_sd=0.1,
        adstock_max_lag=2,
        saturation_alpha_mean=2.5,
        saturation_alpha_sd=1.0,
        saturation_lam_mean=0.4,
        saturation_lam_sd=0.2,
        beta_mean=0.2,
        beta_sd=0.12,
        notes="High-intent search channel. Short carryover effect. "
              "Often captures demand created by other channels."
    ),

    "tiktok": ChannelPrior(
        # TikTok: Discovery-driven, moderate carryover.
        # Content can go viral and have delayed effects.
        adstock_decay_mean=0.35,
        adstock_decay_sd=0.15,
        adstock_max_lag=3,
        saturation_alpha_mean=1.8,
        saturation_alpha_sd=1.0,
        saturation_lam_mean=0.5,
        saturation_lam_sd=0.3,
        beta_mean=0.12,
        beta_sd=0.08,
        notes="Discovery platform. Content-driven. Moderate lag effect "
              "as awareness converts over days/weeks."
    ),

    "pinterest": ChannelPrior(
        # Pinterest: Long consideration cycle, high carryover.
        # Users save pins and come back later to purchase.
        adstock_decay_mean=0.55,
        adstock_decay_sd=0.15,
        adstock_max_lag=6,
        saturation_alpha_mean=1.5,
        saturation_alpha_sd=0.8,
        saturation_lam_mean=0.6,
        saturation_lam_sd=0.3,
        beta_mean=0.08,
        beta_sd=0.06,
        notes="Upper-funnel discovery. Long carryover — users pin and return "
              "weeks later. Lower direct attribution but persistent effect."
    ),

    "sms": ChannelPrior(
        # SMS: Immediate response, near-zero carryover.
        # People act on SMS within hours, not weeks.
        adstock_decay_mean=0.05,
        adstock_decay_sd=0.05,
        adstock_max_lag=1,
        saturation_alpha_mean=3.0,
        saturation_alpha_sd=1.0,
        saturation_lam_mean=0.3,
        saturation_lam_sd=0.2,
        beta_mean=0.1,
        beta_sd=0.08,
        notes="Direct response channel. Nearly immediate effect. "
              "Primarily drives returning customer revenue."
    ),

    "email": ChannelPrior(
        # Email: Short carryover (1-2 days after send), but consistent.
        # Measured by number of emails sent per week.
        adstock_decay_mean=0.1,
        adstock_decay_sd=0.08,
        adstock_max_lag=1,
        saturation_alpha_mean=2.5,
        saturation_alpha_sd=1.0,
        saturation_lam_mean=0.5,
        saturation_lam_sd=0.3,
        beta_mean=0.1,
        beta_sd=0.07,
        notes="Retention/CRM channel. Short lag (open within days). "
              "Primarily drives returning customer revenue."
    ),

    "snapchat": ChannelPrior(
        # Snapchat: Similar to TikTok but smaller effect for most DTC.
        adstock_decay_mean=0.3,
        adstock_decay_sd=0.15,
        adstock_max_lag=3,
        saturation_alpha_mean=1.8,
        saturation_alpha_sd=1.0,
        saturation_lam_mean=0.5,
        saturation_lam_sd=0.3,
        beta_mean=0.08,
        beta_sd=0.06,
        notes="Younger demographic. Similar to TikTok but typically "
              "smaller scale for DTC brands."
    ),
}

# Priors for control variables (not media channels)
CONTROL_PRIORS = {
    "discount_campaign": {
        "mean": 0.3,
        "sd": 0.2,
        "notes": "Discounts typically lift revenue 15-50% during campaign weeks"
    },
    "product_drop": {
        "mean": 0.25,
        "sd": 0.15,
        "notes": "New product drops drive excitement and purchase; effect is immediate"
    },
    "holiday": {
        "mean": 0.2,
        "sd": 0.15,
        "notes": "Shopping holidays (BF, Christmas) naturally lift revenue"
    },
}


def get_channel_prior(channel_name: str) -> ChannelPrior:
    """Get prior for a channel, with sensible defaults for unknown channels."""
    # Normalize name
    clean = channel_name.lower().replace(" ", "_").replace("-", "_")

    if clean in DTC_CHANNEL_PRIORS:
        return DTC_CHANNEL_PRIORS[clean]

    # Default prior for unknown channels
    return ChannelPrior(
        adstock_decay_mean=0.3,
        adstock_decay_sd=0.15,
        adstock_max_lag=4,
        saturation_alpha_mean=2.0,
        saturation_alpha_sd=1.0,
        saturation_lam_mean=0.5,
        saturation_lam_sd=0.3,
        beta_mean=0.1,
        beta_sd=0.08,
        notes=f"Default prior for unknown channel: {channel_name}"
    )
