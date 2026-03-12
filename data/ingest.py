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
    ingester = WindsorIngester(api_key=api_key)
    result = {}

    # Fetch ad platform data
    for channel_name, channel_cfg in client_cfg.get("channels", {}).items():
        if channel_cfg is None:
            continue

        account_id = channel_cfg.get("windsor_account")
        if not account_id:
            logger.info(f"Skipping {channel_name}: no account configured")
            continue

        try:
            df = ingester.fetch_channel_data(
                connector=channel_cfg["windsor_connector"],
                account_id=account_id,
                date_from=date_from,
                date_to=date_to,
                fields=[
                    "date",
                    channel_cfg.get("spend_field", "spend"),
                    channel_cfg.get("impressions_field", "impressions"),
                ],
            )
            # Normalize column names
            df = df.rename(columns={
                channel_cfg.get("spend_field", "spend"): "spend",
                channel_cfg.get("impressions_field", "impressions"): "impressions",
            })
            result[channel_name] = df
        except Exception as e:
            logger.error(f"Failed to fetch {channel_name} for {client_key}: {e}")
            result[channel_name] = pd.DataFrame(columns=["date", "spend", "impressions"])

    # Fetch revenue data from Shopify — aggregated directly to WEEKLY.
    # Shopify returns one row per order, so large date ranges exceed Windsor's
    # response size limit. We fetch in monthly chunks with year_week_iso and
    # aggregate to weekly totals. Note: customer_is_returning causes size limit
    # errors even for quarterly ranges, so we skip it for now.
    rev_cfg = client_cfg.get("revenue_source", {})
    rev_account = rev_cfg.get("windsor_account")
    if rev_account:
        all_shopify_chunks = []
        start = pd.Timestamp(date_from)
        end = pd.Timestamp(date_to)
        # Monthly chunks to stay safely under Windsor's size limit
        chunk_starts = pd.date_range(start, end, freq="MS")
        if len(chunk_starts) == 0:
            chunk_starts = pd.DatetimeIndex([start])

        for i, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + pd.offsets.MonthEnd(1), end)
            try:
                df_chunk = ingester.fetch_channel_data(
                    connector=rev_cfg["windsor_connector"],
                    account_id=rev_account,
                    date_from=str(chunk_start.date()),
                    date_to=str(chunk_end.date()),
                    fields=["year_week_iso", "order_total_price", "order_count"],
                )
                if not df_chunk.empty:
                    all_shopify_chunks.append(df_chunk)
                    logger.info(f"  Shopify month {i+1}/{len(chunk_starts)}: {len(df_chunk)} rows")
            except Exception as e:
                logger.error(f"  Shopify month {chunk_start.date()} failed: {e}")

        if all_shopify_chunks:
            df = pd.concat(all_shopify_chunks, ignore_index=True)

            # Convert year_week_iso ("2025|03") to a Monday date for that ISO week
            def yw_to_date(yw: str) -> pd.Timestamp:
                year, week = yw.split("|")
                return pd.Timestamp.fromisocalendar(int(year), int(week), 1)

            df["week_start"] = df["year_week_iso"].apply(yw_to_date)

            # Aggregate directly to weekly totals
            shopify_df = df.groupby("week_start").agg(
                revenue=("order_total_price", "sum"),
                orders=("order_count", "sum"),
            ).reset_index()

            result["shopify"] = shopify_df
            logger.info(f"  Shopify total: {len(shopify_df)} weekly rows")
        else:
            logger.error(f"No Shopify data retrieved for {client_key}")
            result["shopify"] = pd.DataFrame(
                columns=["week_start", "revenue", "orders"]
            )
    else:
        logger.warning(f"No Shopify account configured for {client_key}")

    # Fetch email data from Klaviyo (via Windsor) if configured
    # Klaviyo returns one row per campaign/flow per day.
    # We aggregate to daily totals (opens, clicks, revenue).
    # Opens serve as the "reach" proxy since Windsor/Klaviyo doesn't expose sends.
    email_cfg = client_cfg.get("email_source", {})
    email_account = email_cfg.get("windsor_account")
    if email_account:
        try:
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
