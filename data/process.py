"""
Data processing pipeline: daily -> weekly aggregation, channel merging.

Takes raw per-channel DataFrames from ingest.py and produces a single
model-ready DataFrame with one row per week.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def daily_to_weekly(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Aggregate daily data to weekly (Monday-start weeks).

    Sums all numeric columns within each ISO week.
    """
    if df.empty:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Resample to weekly (Monday start), summing all numeric columns
    df = df.set_index(date_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    weekly = df[numeric_cols].resample("W-MON", label="left", closed="left").sum()
    weekly = weekly.reset_index()
    weekly = weekly.rename(columns={date_col: "week_start"})

    return weekly


def merge_channel_data(
    raw_data: dict[str, pd.DataFrame],
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """
    Merge all channel DataFrames into a single weekly DataFrame.

    Args:
        raw_data: Dict from fetch_client_data() with keys like 'meta', 'google_ads', 'shopify'
        date_from: Start date for the complete date range
        date_to: End date for the complete date range

    Returns:
        DataFrame with columns:
            week_start, revenue, orders,
            meta_spend, meta_impressions,
            google_ads_spend, google_ads_impressions,
            pinterest_spend, pinterest_impressions,
            tiktok_spend, tiktok_impressions,
            sms_spend,
            ... (any other configured channels)
    """
    # Create complete weekly date range
    all_weeks = pd.date_range(
        start=pd.Timestamp(date_from) - pd.Timedelta(days=pd.Timestamp(date_from).weekday()),
        end=date_to,
        freq="W-MON",
    )
    result = pd.DataFrame({"week_start": all_weeks})

    # Process revenue (Shopify)
    # Shopify data arrives pre-aggregated to weekly (with week_start column)
    # from ingest.py. If it still has a "date" column, fall back to daily→weekly.
    if "shopify" in raw_data and not raw_data["shopify"].empty:
        shopify_df = raw_data["shopify"]
        logger.info(f"Shopify data: {len(shopify_df)} rows, cols={list(shopify_df.columns)}")
        if "week_start" in shopify_df.columns:
            shopify_weekly = shopify_df.copy()
            shopify_weekly["week_start"] = pd.to_datetime(shopify_weekly["week_start"])
        else:
            shopify_weekly = daily_to_weekly(shopify_df)

        rev_cols = [c for c in shopify_weekly.columns
                    if c != "week_start" and c in (
                        "revenue", "orders", "new_revenue", "new_orders",
                        "returning_revenue", "returning_orders", "total_discounts",
                    )]
        if rev_cols:
            merge_cols = ["week_start"] + rev_cols
            result = result.merge(
                shopify_weekly[merge_cols].drop_duplicates("week_start"),
                on="week_start",
                how="left",
            )
            logger.info(f"After Shopify merge: revenue col exists={('revenue' in result.columns)}, non-null={result.get('revenue', pd.Series()).notna().sum()}")
    else:
        logger.warning(f"Shopify data is empty or missing. raw_data keys={list(raw_data.keys())}, "
                       f"shopify empty={raw_data.get('shopify', pd.DataFrame()).empty}")

    # Ensure revenue and orders columns always exist (even if Shopify data was empty)
    if "revenue" not in result.columns:
        logger.warning("Revenue column missing after Shopify merge — adding zeros")
        result["revenue"] = 0.0
    if "orders" not in result.columns:
        result["orders"] = 0.0

    # Process each ad channel (paid media)
    ad_channels = [k for k in raw_data if k not in ("shopify", "sms", "email")]
    for channel_name in ad_channels:
        df = raw_data[channel_name]
        if df.empty:
            result[f"{channel_name}_spend"] = 0.0
            result[f"{channel_name}_impressions"] = 0.0
            continue

        weekly = daily_to_weekly(df)

        # Rename columns with channel prefix
        rename_map = {}
        for col in weekly.columns:
            if col == "week_start":
                continue
            rename_map[col] = f"{channel_name}_{col}"
        weekly = weekly.rename(columns=rename_map)

        # Merge
        result = result.merge(weekly, on="week_start", how="left")

    # Process email/Klaviyo data (uses opens as the media variable, not spend)
    # Email opens serve as the "volume" metric since Klaviyo doesn't have a spend metric.
    # In the MMM, email_opens is treated like an impression/reach variable.
    if "email" in raw_data and not raw_data["email"].empty:
        email_weekly = daily_to_weekly(raw_data["email"])
        # email_opens becomes the model input; email_clicks and email_revenue kept for reference
        result = result.merge(email_weekly, on="week_start", how="left")
        for col in ["email_opens", "email_clicks", "email_revenue"]:
            if col in result.columns:
                result[col] = result[col].fillna(0)

    # Process SMS (already weekly from CSV)
    if "sms" in raw_data and not raw_data["sms"].empty:
        sms_df = raw_data["sms"].copy()
        if "date" in sms_df.columns:
            sms_df = sms_df.rename(columns={"date": "week_start"})
        sms_df["week_start"] = pd.to_datetime(sms_df["week_start"])
        sms_rename = {c: f"sms_{c}" for c in sms_df.columns if c != "week_start"}
        sms_df = sms_df.rename(columns=sms_rename)
        result = result.merge(sms_df, on="week_start", how="left")

    # Fill NaN spend/impression columns with 0
    fill_cols = [c for c in result.columns if c.endswith(("_spend", "_impressions"))]
    result[fill_cols] = result[fill_cols].fillna(0)

    # Fill NaN revenue with 0 (weeks with no orders)
    if "revenue" in result.columns:
        result["revenue"] = result["revenue"].fillna(0)
    if "orders" in result.columns:
        result["orders"] = result["orders"].fillna(0)

    # Sort by date
    result = result.sort_values("week_start").reset_index(drop=True)

    # Drop weeks where we have no revenue data at all (before Shopify was connected)
    if "revenue" in result.columns:
        # Keep only weeks where we have SOME data
        has_any_data = result.drop(columns=["week_start"]).sum(axis=1) > 0
        result = result[has_any_data].reset_index(drop=True)

    logger.info(f"Merged dataset: {len(result)} weeks, {len(result.columns)} columns")
    logger.info(f"  Date range: {result['week_start'].min()} to {result['week_start'].max()}")
    logger.info(f"  Channels: {[c.replace('_spend','') for c in result.columns if c.endswith('_spend')]}")

    return result


def prepare_model_input(
    merged_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Final preparation for model input.

    Adds control variables (events), time features, and validates data quality.

    Returns:
        Model-ready DataFrame
    """
    df = merged_df.copy()

    # Add time features
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["year"] = df["week_start"].dt.year

    # Merge promotional events
    if events_df is not None and not events_df.empty:
        events = events_df.copy()
        events["week_start"] = pd.to_datetime(events["week_start"])
        event_cols = [c for c in events.columns if c != "week_start" and c != "notes"]
        df = df.merge(events[["week_start"] + event_cols], on="week_start", how="left")
        for col in event_cols:
            df[col] = df[col].fillna(0).astype(int)
    else:
        # Add empty event columns
        df["discount_campaign"] = 0
        df["product_drop"] = 0
        df["holiday"] = 0

    # Identify spend columns for the model
    spend_cols = sorted([c for c in df.columns if c.endswith("_spend")])

    # Data quality checks
    n_weeks = len(df)
    if n_weeks < 52:
        logger.warning(f"Only {n_weeks} weeks of data. Minimum 52 recommended, 104 ideal.")
    else:
        logger.info(f"Data quality: {n_weeks} weeks available (good)" if n_weeks >= 104
                     else f"Data quality: {n_weeks} weeks available (acceptable, 104 ideal)")

    # Check for zero-spend channels (can't model these)
    for col in spend_cols:
        total = df[col].sum()
        if total == 0:
            logger.warning(f"Channel {col} has zero total spend - will be excluded from model")
        else:
            nonzero_weeks = (df[col] > 0).sum()
            logger.info(f"Channel {col}: total={total:,.0f}, active {nonzero_weeks}/{n_weeks} weeks")

    return df


def get_media_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of active media input columns for the model.

    This includes:
    - *_spend columns (paid channels: Meta, Google Ads, Pinterest, TikTok, SMS)
    - email_opens (Klaviyo: uses opens as the volume/reach proxy)
    """
    spend_cols = [c for c in df.columns if c.endswith("_spend")]
    media_cols = [c for c in spend_cols if df[c].sum() > 0]

    # Add email_opens as a media variable if present and active
    if "email_opens" in df.columns and df["email_opens"].sum() > 0:
        media_cols.append("email_opens")

    return media_cols


# Backwards compatibility alias
get_spend_columns = get_media_columns


def get_channel_names(media_cols: list[str]) -> list[str]:
    """Convert media column names to clean display names."""
    names = []
    for c in media_cols:
        if c == "email_opens":
            names.append("Email (Klaviyo)")
        else:
            names.append(c.replace("_spend", "").replace("_", " ").title())
    return names
