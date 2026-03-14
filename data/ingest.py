"""
Data ingestion from Windsor.ai REST API.

Windsor.ai provides a unified API to pull marketing data from all connected platforms.
Docs: https://windsor.ai/api-fields/

Usage:
    ingester = WindsorIngester(api_key="your-key")
    meta_data = ingester.fetch_channel_data(
        connector="facebook",
        account_id="260486998006445",
        date_from="2024-01-01",
        date_to="2026-03-01",
        fields=["date", "spend", "impressions", "clicks"]
    )
"""

import os
import requests
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WindsorIngester:
    """Pull marketing data from Windsor.ai's REST API."""

    BASE_URL = "https://connectors.windsor.ai/all"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("WINDSOR_API_KEY")

        # Also try Streamlit secrets if available
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("windsor", {}).get("api_key", "")
            except Exception:
                pass

        if not self.api_key:
            raise ValueError(
                "Windsor API key required. Set it in .streamlit/secrets.toml, "
                "as WINDSOR_API_KEY env var, or pass api_key parameter."
            )

    def fetch_channel_data(
        self,
        connector: str,
        account_id: str,
        date_from: str,
        date_to: str,
        fields: list[str],
    ) -> pd.DataFrame:
        """
        Fetch data from a specific Windsor connector/account.

        Args:
            connector: Windsor connector name (e.g., 'facebook', 'google_ads', 'shopify')
            account_id: The account ID within that connector
            date_from: Start date 'YYYY-MM-DD'
            date_to: End date 'YYYY-MM-DD'
            fields: List of field names to retrieve

        Returns:
            DataFrame with requested fields
        """
        params = {
            "api_key": self.api_key,
            "connector": connector,
            "account_id": account_id,
            "date_from": date_from,
            "date_to": date_to,
            "fields": ",".join(fields),
        }

        logger.info(f"Fetching {connector} data for account {account_id} ({date_from} to {date_to})")

        response = requests.get(self.BASE_URL, params=params, timeout=120)
        response.raise_for_status()

        data = response.json()

        if "data" in data:
            df = pd.DataFrame(data["data"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unexpected Windsor API response format: {list(data.keys())}")

        if df.empty:
            logger.warning(f"No data returned for {connector}/{account_id}")
            return pd.DataFrame(columns=fields)

        # Ensure date column exists and is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Convert numeric columns (skip string-type columns like year_week_iso)
        skip_cols = {"date", "year_week_iso", "subject"}
        numeric_cols = [c for c in df.columns if c not in skip_cols]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        logger.info(f"  -> Got {len(df)} rows")
        return df


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the platform configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_client_data(
    client_key: str,
    config: dict,
    date_from: str,
    date_to: str,
    api_key: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all channel data for a specific client.

    Returns a dict of DataFrames:
        {
            "meta": DataFrame with date/spend/impressions,
            "google_ads": DataFrame with date/spend/impressions,
            "pinterest": ...,
            "shopify": DataFrame with date/revenue/orders,
            "sms": DataFrame from CSV (if exists),
        }
    """
    client_cfg = config["clients"][client_key]
    result = {}

    # Load ad platform data from pre-built weekly CSVs.
    # Like Shopify, ad data is pre-aggregated offline and committed to the repo
    # so the deployed app doesn't need a Windsor API key at runtime.
    for channel_name in client_cfg.get("channels", {}):
        channel_csv = Path(__file__).parent / f"{client_key}_{channel_name}_weekly.csv"
        logger.info(f"  {channel_name} CSV path: {channel_csv} (exists={channel_csv.exists()})")

        if channel_csv.exists():
            ch_df = pd.read_csv(channel_csv, parse_dates=["week_start"])
            logger.info(f"  {channel_name} CSV raw: {len(ch_df)} rows")
            dt_from = pd.Timestamp(date_from)
            dt_to = pd.Timestamp(date_to)
            ch_df = ch_df[
                (ch_df["week_start"] >= dt_from)
                & (ch_df["week_start"] <= dt_to)
            ].copy()
            result[channel_name] = ch_df
            logger.info(f"  {channel_name}: {len(ch_df)} weekly rows after date filter")
        else:
            logger.warning(f"No {channel_name} CSV found at {channel_csv}")
            result[channel_name] = pd.DataFrame(columns=["week_start", "spend", "impressions", "clicks"])

    # Load revenue data from pre-built CSV.
    # Revenue is sourced from a Shopify analytics export (not Windsor) and
    # pre-processed into weekly gross sales (net sales + returns) with
    # new/returning customer splits.
    # To update: upload new/returning CSVs via the dashboard.
    shopify_csv = Path(__file__).parent / f"{client_key}_shopify_weekly.csv"
    logger.info(f"  Revenue CSV path: {shopify_csv} (exists={shopify_csv.exists()})")
    if shopify_csv.exists():
        shopify_df = pd.read_csv(shopify_csv, parse_dates=["week_start"])
        logger.info(f"  Revenue CSV: {len(shopify_df)} rows, cols={list(shopify_df.columns)}")
        dt_from = pd.Timestamp(date_from)
        dt_to = pd.Timestamp(date_to)
        shopify_df = shopify_df[
            (shopify_df["week_start"] >= dt_from)
            & (shopify_df["week_start"] <= dt_to)
        ].copy()
        result["shopify"] = shopify_df
        logger.info(f"  Revenue: {len(shopify_df)} weekly rows after date filter ({dt_from.date()} to {dt_to.date()})")
    else:
        logger.warning(f"No revenue CSV found at {shopify_csv}. Upload revenue data via the dashboard.")
        result["shopify"] = pd.DataFrame(columns=["week_start", "revenue", "new_revenue", "returning_revenue", "orders"])

    # Fetch email data from Klaviyo (via Windsor) if configured
    # Klaviyo returns one row per campaign/flow per day.
    # We aggregate to daily totals (opens, clicks, revenue).
    # Opens serve as the "reach" proxy since Windsor/Klaviyo doesn't expose sends.
    email_cfg = client_cfg.get("email_source", {})
    email_account = email_cfg.get("windsor_account")
    if email_account:
        try:
            ingester = WindsorIngester(api_key=api_key)
            df = ingester.fetch_channel_data(
                connector=email_cfg["windsor_connector"],
                account_id=email_account,
                date_from=date_from,
                date_to=date_to,
                fields=[
                    "date",
                    email_cfg.get("opens_field", "opens"),
                    email_cfg.get("clicks_field", "clicks"),
                    email_cfg.get("revenue_field", "revenue"),
                ],
            )
            # Aggregate across all campaigns/flows per day
            if not df.empty and "date" in df.columns:
                df = df.groupby("date").agg(
                    email_opens=(email_cfg.get("opens_field", "opens"), "sum"),
                    email_clicks=(email_cfg.get("clicks_field", "clicks"), "sum"),
                    email_revenue=(email_cfg.get("revenue_field", "revenue"), "sum"),
                ).reset_index()
            else:
                df = pd.DataFrame(columns=["date", "email_opens", "email_clicks", "email_revenue"])

            result["email"] = df
            logger.info(f"  Email/Klaviyo data: {len(df)} daily rows")
        except Exception as e:
            logger.warning(f"Could not fetch Klaviyo data: {e}. Email will be excluded from model.")

    # Load SMS spend from CSV if available
    sms_csv = client_cfg.get("sms_csv")
    if sms_csv and Path(sms_csv).exists():
        logger.info(f"Loading SMS data from {sms_csv}")
        sms_df = pd.read_csv(sms_csv, parse_dates=["week_start"])
        sms_df = sms_df.rename(columns={"week_start": "date"})
        result["sms"] = sms_df

    return result


def process_revenue_csvs(
    new_customers_df: pd.DataFrame,
    returning_customers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process Shopify analytics export CSVs (new + returning customers) into
    weekly gross revenue (net sales + returns added back).

    Expected input columns: Week, Total returns, Net sales
    (as exported from Shopify analytics "Net Sales Over Time" report)

    Returns DataFrame with: week_start, revenue, new_revenue, returning_revenue
    """
    new = new_customers_df[["Week", "Total returns", "Net sales"]].copy()
    ret = returning_customers_df[["Week", "Total returns", "Net sales"]].copy()

    # Gross sales = net sales + abs(returns).  Returns are negative, so subtract.
    new["new_revenue"] = new["Net sales"] - new["Total returns"].fillna(0)
    ret["returning_revenue"] = ret["Net sales"] - ret["Total returns"].fillna(0)

    combined = new[["Week", "new_revenue"]].merge(
        ret[["Week", "returning_revenue"]], on="Week", how="outer"
    ).sort_values("Week").reset_index(drop=True).fillna(0)

    combined["revenue"] = combined["new_revenue"] + combined["returning_revenue"]
    combined = combined.rename(columns={"Week": "week_start"})
    combined["week_start"] = pd.to_datetime(combined["week_start"])

    logger.info(f"Processed revenue CSVs: {len(combined)} weeks, "
                f"total revenue={combined['revenue'].sum():,.0f}")
    return combined[["week_start", "revenue", "new_revenue", "returning_revenue"]]
