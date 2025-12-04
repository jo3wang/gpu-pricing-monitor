#!/usr/bin/env python3
"""
AWS Spot Price Fallback Generator
Generates estimated spot pricing data when API access is unavailable.
Uses historical average spot prices as estimates for project deadline requirements.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import pandas as pd

from utils import detect_aws_accel_model_and_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Target GPU instance types
TARGET_GPU_INSTANCES = [
    # P3 family (V100)
    "p3.2xlarge", "p3.8xlarge", "p3.16xlarge", "p3dn.24xlarge",

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
]

TARGET_REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]


def generate_estimated_spot_prices(region: str) -> List[Dict]:
    """
    Alternative approach: Use AWS Instance Types API (public, no auth required).
    This provides instance specifications but not real-time spot prices.

    For a 48-hour deadline, we'll use historical average spot prices as estimates.

    Args:
        region: AWS region code

    Returns:
        List of spot pricing records with estimated prices
    """
    logging.info(f"Using estimated spot prices for {region} (fallback method)")

    # Historical average spot prices for GPU instances (as of Dec 2024)
    # These are approximations based on typical spot pricing
    ESTIMATED_SPOT_PRICES = {
        # P3 family (V100) - typically 30-50% of on-demand
        "p3.2xlarge": 0.90,
        "p3.8xlarge": 3.60,
        "p3.16xlarge": 7.20,
        "p3dn.24xlarge": 9.50,

        # P4 family (A100)
        "p4d.24xlarge": 12.00,
        "p4de.24xlarge": 15.00,

        # P5 family (H100)
        "p5.48xlarge": 30.00,

        # G4 family (T4)
        "g4dn.xlarge": 0.16,
        "g4dn.2xlarge": 0.23,
        "g4dn.4xlarge": 0.36,
        "g4dn.12xlarge": 1.16,

        # G5 family (A10G)
        "g5.xlarge": 0.27,
        "g5.2xlarge": 0.41,
        "g5.4xlarge": 0.65,
        "g5.12xlarge": 1.95,

        # G6 family (L4)
        "g6.xlarge": 0.35,
        "g6.2xlarge": 0.55,
        "g6.4xlarge": 0.88,
        "g6.12xlarge": 2.64,
    }

    records = []
    timestamp_utc = datetime.utcnow().isoformat() + 'Z'

    for instance_type in TARGET_GPU_INSTANCES:
        accelerator_model, accelerator_count = detect_aws_accel_model_and_count(instance_type)

        if not accelerator_model:
            continue

        if instance_type not in ESTIMATED_SPOT_PRICES:
            logging.warning(f"No estimated price for {instance_type}, skipping")
            continue

        spot_price = ESTIMATED_SPOT_PRICES[instance_type]
        price_per_accel = spot_price / accelerator_count

        record = {
            'timestamp_utc': timestamp_utc,
            'cloud': 'aws',
            'pricing_type': 'spot_estimated',  # Mark as estimated
            'region': region,
            'instance_type': instance_type,
            'accelerator_type': 'GPU',
            'accelerator_model': accelerator_model,
            'accelerator_count': accelerator_count,
            'price_hour_spot': spot_price,
            'price_per_accel_hour_spot': price_per_accel,
            'data_source': 'estimated_from_historical_averages',
            'note': 'Estimated prices - API access pending quota approval'
        }

        records.append(record)
        logging.info(f"  {instance_type}: ${spot_price:.3f}/hr (${price_per_accel:.3f}/GPU/hr)")

    return records


def main():
    """Main function to generate estimated AWS spot pricing."""
    logging.info("=" * 80)
    logging.info("AWS Spot Price Fallback Generator")
    logging.info("Using estimated historical spot prices (API unavailable)")
    logging.info("=" * 80)

    all_records = []

    for region in TARGET_REGIONS:
        try:
            logging.info(f"\nGenerating spot prices for {region}...")
            records = generate_estimated_spot_prices(region)
            all_records.extend(records)

            logging.info(f"Generated {len(records)} estimated spot prices for {region}")

        except Exception as e:
            logging.error(f"Failed to generate spot data for {region}: {e}")
            continue

    if not all_records:
        logging.error("No spot pricing data collected")
        return

    # Convert to DataFrame
    df_spot = pd.DataFrame(all_records)

    # Save to CSV
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    output_path = data_dir / f"aws_gpu_spot_prices_estimated_{date_str}.csv"

    df_spot.to_csv(output_path, index=False)

    logging.info("=" * 80)
    logging.info(f"Saved {len(df_spot)} estimated spot price records → {output_path}")
    logging.info("=" * 80)

    # Summary statistics
    logging.info("\nSummary by region:")
    for region in df_spot['region'].unique():
        region_df = df_spot[df_spot['region'] == region]
        models = sorted(region_df['accelerator_model'].unique())
        price_range = region_df['price_per_accel_hour_spot']
        logging.info(f"  {region}: {len(region_df)} instances")
        logging.info(f"    Models: {', '.join(models)}")
        logging.info(f"    Price range: ${price_range.min():.3f}-${price_range.max():.3f} per GPU/hr")

    logging.info("\n" + "=" * 80)
    logging.info("NOTE: These are estimated prices based on historical spot pricing")
    logging.info("Once AWS approves your quota requests, run fetch_aws_spot_gpu.py")
    logging.info("to get real-time spot pricing data from the API.")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
