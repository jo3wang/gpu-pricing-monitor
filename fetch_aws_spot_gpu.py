#!/usr/bin/env python3
"""
AWS EC2 Spot GPU Pricing Monitor
Fetches spot pricing for GPU instances using boto3 and saves to CSV.
Collects historical data for the past 7 days, saved as separate daily CSVs.
"""

import boto3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from collections import defaultdict

# --- AWS Spot GPU collection ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Import shared utilities
from utils import detect_aws_accel_model_and_count, save_to_csv

# Target GPU instance types for spot monitoring
TARGET_GPU_INSTANCES = [
    # P3 family (V100)
    "p3.2xlarge", "p3.8xlarge", "p3.16xlarge",
    "p3dn.24xlarge",

    # P4 family (A100)
    "p4d.24xlarge", "p4de.24xlarge",

    # P5 family (H100)
    "p5.48xlarge",

    # G4 family (T4)
    "g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge", "g4dn.12xlarge",

    # G5 family (A10G)
    "g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.12xlarge",

    # G6 family (L4)
    "g6.xlarge", "g6.2xlarge", "g6.4xlarge", "g6.12xlarge",

    # G6e family (L40S)
    "g6e.xlarge", "g6e.2xlarge", "g6e.4xlarge", "g6e.12xlarge",
]

# Target regions for spot monitoring
TARGET_REGIONS = [
    "us-east-1",
    "us-west-2",
    "eu-west-1"
]

# Spot history lookback window (days)
LOOKBACK_DAYS = 7


def validate_spot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans spot pricing DataFrame.

    Args:
        df: Raw spot pricing DataFrame

    Returns:
        Validated DataFrame
    """
    if df.empty:
        logging.warning("Empty DataFrame provided to validate_spot_df")
        return df

    # Required columns for spot data
    required_cols = [
        'timestamp_utc', 'cloud', 'pricing_type', 'region', 'instance_type',
        'accelerator_type', 'accelerator_model', 'accelerator_count',
        'price_hour_spot', 'price_per_accel_hour_spot'
    ]

    # Check all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in spot data: {missing_cols}")

    logging.info("aws-spot: Schema validation passed - all required columns present")

    # Value validation
    initial_rows = len(df)

    # Remove rows with invalid counts, prices, or null values
    df_clean = df[
        (df['accelerator_count'] > 0) &
        (df['price_hour_spot'] > 0) &
        (df['price_per_accel_hour_spot'] > 0) &
        (df['accelerator_model'].notna()) &
        (df['instance_type'].notna())
    ].copy()

    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        logging.warning(f"aws-spot: Dropped {dropped_rows} invalid rows (invalid counts, prices, or null values)")

    logging.info(f"aws-spot: Value validation passed - {len(df_clean)} valid rows remaining")

    return df_clean


def fetch_spot_prices_region(region: str, lookback_days: int = 7) -> Dict[str, List[Dict]]:
    """
    Fetch spot price history for GPU instances in a specific region.
    Returns data grouped by date.

    Args:
        region: AWS region code (e.g., 'us-east-1')
        lookback_days: Days to look back for price history

    Returns:
        Dictionary mapping date strings to lists of spot price records
    """
    logging.info(f"Fetching spot prices for region: {region}")

    try:
        ec2 = boto3.client("ec2", region_name=region)
    except Exception as e:
        logging.error(f"Failed to create EC2 client for region {region}: {e}")
        return {}

    # Calculate time window
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)

    logging.info(f"Fetching spot history from {start_time.date()} to {end_time.date()}")

    # Group records by date
    records_by_date = defaultdict(list)

    for instance_type in TARGET_GPU_INSTANCES:
        # Check if this is a GPU instance we care about
        accelerator_model, accelerator_count = detect_aws_accel_model_and_count(instance_type)
        if not accelerator_model:
            continue

        try:
            logging.info(f"Querying spot prices for {instance_type} in {region}")

            # Query spot price history - AWS returns newest first
            next_token = None
            all_spot_prices = []

            while True:
                params = {
                    'InstanceTypes': [instance_type],
                    'ProductDescriptions': ["Linux/UNIX"],
                    'StartTime': start_time,
                    'EndTime': end_time,
                    'MaxResults': 1000  # Max allowed by API
                }

                if next_token:
                    params['NextToken'] = next_token

                response = ec2.describe_spot_price_history(**params)

                spot_prices = response.get('SpotPrices', [])
                all_spot_prices.extend(spot_prices)

                next_token = response.get('NextToken')
                if not next_token:
                    break

            if not all_spot_prices:
                logging.warning(f"No spot price data found for {instance_type} in {region}")
                continue

            logging.info(f"Retrieved {len(all_spot_prices)} spot price records for {instance_type}")

            # Group prices by date and take the median price per day
            prices_by_date = defaultdict(list)
            for price_point in all_spot_prices:
                spot_timestamp = price_point['Timestamp']
                date_str = spot_timestamp.strftime('%Y-%m-%d')
                spot_price = float(price_point['SpotPrice'])
                prices_by_date[date_str].append({
                    'spot_price': spot_price,
                    'availability_zone': price_point['AvailabilityZone'],
                    'timestamp': spot_timestamp
                })

            # For each date, calculate median spot price
            for date_str, prices_list in prices_by_date.items():
                prices = [p['spot_price'] for p in prices_list]
                median_spot_price = np.median(prices)
                price_per_accel_hour = median_spot_price / accelerator_count

                # Get the most recent timestamp for this date
                most_recent = max(prices_list, key=lambda x: x['timestamp'])

                record = {
                    'timestamp_utc': datetime.strptime(date_str, '%Y-%m-%d').isoformat() + 'Z',
                    'cloud': 'aws',
                    'pricing_type': 'spot',
                    'region': region,
                    'instance_type': instance_type,
                    'accelerator_type': 'GPU',
                    'accelerator_model': accelerator_model,
                    'accelerator_count': accelerator_count,
                    'price_hour_spot': median_spot_price,
                    'price_per_accel_hour_spot': price_per_accel_hour,
                    'availability_zone': most_recent['availability_zone'],
                    'spot_timestamp': most_recent['timestamp'].isoformat() + 'Z',
                    'num_samples': len(prices)
                }

                records_by_date[date_str].append(record)

            logging.info(f"Processed {instance_type}: {len(prices_by_date)} days of data")

        except Exception as e:
            logging.error(f"Error fetching spot price for {instance_type} in {region}: {e}")
            continue

    logging.info(f"Collected spot prices for {len(records_by_date)} dates from {region}")
    return records_by_date


def main():
    """Main function to collect AWS spot GPU pricing data."""
    logging.info("=" * 80)
    logging.info("Starting AWS Spot GPU pricing collection")
    logging.info(f"Target regions: {TARGET_REGIONS}")
    logging.info(f"Target instance types: {len(TARGET_GPU_INSTANCES)} GPU instances")
    logging.info(f"Lookback window: {LOOKBACK_DAYS} days")
    logging.info("=" * 80)

    # Collect data grouped by date
    all_records_by_date = defaultdict(list)

    # Collect spot data from each region
    for region in TARGET_REGIONS:
        try:
            region_records_by_date = fetch_spot_prices_region(region, LOOKBACK_DAYS)

            # Merge into all_records_by_date
            for date_str, records in region_records_by_date.items():
                all_records_by_date[date_str].extend(records)

        except Exception as e:
            logging.error(f"Failed to collect spot data from {region}: {e}")
            continue

    if not all_records_by_date:
        logging.error("No spot pricing data collected from any region")
        return

    # Save one CSV per date
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    saved_files = []

    for date_str in sorted(all_records_by_date.keys()):
        records = all_records_by_date[date_str]

        # Convert to DataFrame
        df_spot = pd.DataFrame(records)

        # Validate data
        try:
            df_spot = validate_spot_df(df_spot)
        except Exception as e:
            logging.error(f"Spot data validation failed for {date_str}: {e}")
            continue

        if df_spot.empty:
            logging.warning(f"No valid spot data for {date_str}")
            continue

        # Save to dated CSV
        output_path = data_dir / f"aws_gpu_spot_prices_{date_str}.csv"
        df_spot.to_csv(output_path, index=False)
        saved_files.append(output_path)

        logging.info(f"Saved {len(df_spot)} spot records for {date_str} → {output_path}")

    # Summary statistics
    logging.info("=" * 80)
    logging.info("AWS Spot GPU pricing collection complete")
    logging.info(f"Saved {len(saved_files)} daily CSV files")

    # Overall summary
    total_records = sum(len(all_records_by_date[date_str]) for date_str in all_records_by_date)
    logging.info(f"Total spot records collected: {total_records}")

    # Date range summary
    if all_records_by_date:
        dates = sorted(all_records_by_date.keys())
        logging.info(f"Date range: {dates[0]} to {dates[-1]}")

        # Sample breakdown for latest date
        latest_date = dates[-1]
        latest_df = pd.DataFrame(all_records_by_date[latest_date])

        logging.info(f"\nLatest date ({latest_date}) breakdown:")
        for region in latest_df['region'].unique():
            region_count = len(latest_df[latest_df['region'] == region])
            region_models = sorted(latest_df[latest_df['region'] == region]['accelerator_model'].unique())
            price_range = latest_df[latest_df['region'] == region]['price_per_accel_hour_spot']
            logging.info(f"  {region}: {region_count} instances, models: {', '.join(region_models)}, "
                        f"price range: ${price_range.min():.3f}-${price_range.max():.3f}")

        models_found = sorted(latest_df['accelerator_model'].unique())
        logging.info(f"\nAWS accelerator models with spot pricing: {models_found}")

    logging.info("=" * 80)


if __name__ == "__main__":
    main()
