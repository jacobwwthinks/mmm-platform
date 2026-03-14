"""
Promotional event calendar management.

Handles loading and validating the per-client event CSV files
that track discount campaigns, product drops, and holidays.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Standard holiday dates (can be customized per market)
STANDARD_HOLIDAYS = {
    # Swedish / Nordic market holidays relevant for e-commerce
    "black_friday": lambda year: _get_black_friday(year),
    "cyber_monday": lambda year: _get_black_friday(year) + pd.Timedelta(days=3),
    "christmas_week": lambda year: pd.Timestamp(f"{year}-12-23"),
    "singles_day": lambda year: pd.Timestamp(f"{year}-11-11"),
    "valentines": lambda year: pd.Timestamp(f"{year}-02-14"),
    "midsommar": lambda year: _get_midsommar(year),
}


def _get_black_friday(year: int) -> pd.Timestamp:
    """Black Friday = 4th Friday of November."""
    nov1 = pd.Timestamp(f"{year}-11-01")
    # Find first Friday
    days_until_friday = (4 - nov1.weekday()) % 7
    first_friday = nov1 + pd.Timedelta(days=days_until_friday)
    return first_friday + pd.Timedelta(weeks=3)


def _get_midsommar(year: int) -> pd.Timestamp:
    """Swedish Midsommar = Friday between June 19-25."""
    for day in range(19, 26):
        d = pd.Timestamp(f"{year}-06-{day:02d}")
        if d.weekday() == 4:  # Friday
            return d
    return pd.Timestamp(f"{year}-06-21")  # fallback


def load_events(csv_path: str) -> pd.DataFrame:
    """
    Load event calendar from CSV.

    Expected columns:
        week_start: YYYY-MM-DD (Monday of the week)
        discount_campaign: 0 or 1
        product_drop: 0 or 1
        holiday: 0 or 1
        notes: free text (optional)
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"Events file not found: {csv_path}. Using empty calendar.")
        return pd.DataFrame(columns=["week_start", "discount_campaign", "product_drop", "holiday", "notes"])

    df = pd.read_csv(path, parse_dates=["week_start"])

    # Validate columns
    required = {"week_start", "discount_campaign", "product_drop", "holiday"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {missing}")

    # Ensure integer values (discount_campaign can be 0/1/2 for light/heavy)
    for col in ["discount_campaign", "product_drop", "holiday"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    if "product_offering" in df.columns:
        df["product_offering"] = df["product_offering"].fillna(0).astype(int)

    logger.info(f"Loaded {len(df)} event weeks from {csv_path}")
    logger.info(f"  Discount campaigns: {df['discount_campaign'].sum()} weeks")
    logger.info(f"  Product drops: {df['product_drop'].sum()} weeks")
    logger.info(f"  Holidays: {df['holiday'].sum()} weeks")

    return df


def generate_event_template(
    date_from: str,
    date_to: str,
    output_path: str,
    auto_holidays: bool = True,
) -> pd.DataFrame:
    """
    Generate a blank event template CSV with optional auto-detected holidays.

    Args:
        date_from: Start date
        date_to: End date
        output_path: Where to save the CSV
        auto_holidays: If True, pre-fill known holiday weeks
    """
    # Generate all Monday-start weeks in range
    weeks = pd.date_range(
        start=pd.Timestamp(date_from) - pd.Timedelta(days=pd.Timestamp(date_from).weekday()),
        end=date_to,
        freq="W-MON",
    )

    df = pd.DataFrame({
        "week_start": weeks,
        "discount_campaign": 0,
        "product_drop": 0,
        "holiday": 0,
        "notes": "",
    })

    if auto_holidays:
        for _, week_row in df.iterrows():
            week_date = week_row["week_start"]
            week_end = week_date + pd.Timedelta(days=6)
            year = week_date.year

            for holiday_name, holiday_fn in STANDARD_HOLIDAYS.items():
                try:
                    holiday_date = holiday_fn(year)
                    if week_date <= holiday_date <= week_end:
                        idx = df[df["week_start"] == week_date].index[0]
                        df.loc[idx, "holiday"] = 1
                        existing_notes = df.loc[idx, "notes"]
                        if existing_notes:
                            df.loc[idx, "notes"] = f"{existing_notes}, {holiday_name}"
                        else:
                            df.loc[idx, "notes"] = holiday_name
                except Exception:
                    pass

    df.to_csv(output_path, index=False)
    logger.info(f"Generated event template: {output_path} ({len(df)} weeks)")

    return df
