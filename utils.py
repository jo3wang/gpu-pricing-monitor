#!/usr/bin/env python3
"""
Shared utilities for GPU pricing collectors.
Contains common validation, mapping, and helper functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import logging


# =============================================================================
# AWS GPU Instance Mapping
# =============================================================================

AWS_GPU_INSTANCE_MAP = {
    """Standardized AWS GPU instance mapping for comprehensive coverage."""
    # prefix: (accelerator_model, accelerator_count)
    "p3.": ("V100", 1),
    "p3dn.": ("V100", 8),
    "p4d.": ("A100", 8),
    "p4de.": ("A100", 8),
    "p5.": ("H100", 8),
    "g4dn.": ("T4", 1),
    "g5.": ("A10G", 1),
    "g6.": ("L4", 1),
    "g6e.": ("L40S", 1),
    "p5e.": ("H200", 8),
}


# =============================================================================
# AWS Detection Function
# =============================================================================

def detect_aws_accel_model_and_count(instance_type: str) -> Tuple[Optional[str], int]:
    """
    Maps AWS instance type to accelerator model and count using standardized mapping.

    Args:
        instance_type: AWS instance type (e.g., 'p3.2xlarge', 'g4dn.xlarge')

    Returns:
        Tuple of (accelerator_model, accelerator_count) or (None, 0) if not a GPU instance
    """
    instance_lower = instance_type.lower()

    # Check against our standardized mapping
    for prefix, (model, count) in AWS_GPU_INSTANCE_MAP.items():
        if instance_lower.startswith(prefix):
            return (model, count)

    # Not a GPU instance
    return (None, 0)


# =============================================================================
# Shared Validation Function
# =============================================================================

def validate_df(df: pd.DataFrame, cloud_name: str) -> pd.DataFrame:
    """
    Validates DataFrame schema and data integrity before saving.

    Args:
        df: DataFrame to validate
        cloud_name: Name of cloud provider for logging

    Returns:
        Cleaned and validated DataFrame

    Raises:
        ValueError: If required columns are missing or cloud labels are incorrect
    """
    required_cols = [
        "timestamp_utc",
        "cloud",
        "region",
        "accelerator_type",
        "accelerator_model",
        "accelerator_count",
        "price_per_accel_hour"
    ]

    # Check for missing columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error(f"{cloud_name}: Missing columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    logging.info(f"{cloud_name}: Schema validation passed - all required columns present")

    # Value-level validation
    invalid = df[
        (df["accelerator_count"] <= 0) |
        (~np.isfinite(df["price_per_accel_hour"])) |
        (df["price_per_accel_hour"] <= 0) |
        (df["accelerator_model"].isna()) |
        (df["region"].isna())
    ]

    if len(invalid) > 0:
        logging.warning(f"{cloud_name}: Dropping {len(invalid)} invalid rows (invalid counts, prices, or null values)")
        df = df.drop(invalid.index).reset_index(drop=True)

    # Ensure cloud column is correct
    if not all(df["cloud"] == cloud_name.lower()):
        logging.error(f"{cloud_name}: Incorrect cloud label detected")
        raise ValueError(f"{cloud_name}: Incorrect cloud label detected")

    logging.info(f"{cloud_name}: Value validation passed - {len(df)} valid rows remaining")
    return df


# =============================================================================
# Shared CSV Saving Function
# =============================================================================

def save_to_csv(df: pd.DataFrame, output_dir: Path, filename_prefix: str) -> str:
    """
    Saves DataFrame to CSV file with date-based naming.

    Args:
        df: DataFrame to save
        output_dir: Directory to save CSV in
        filename_prefix: Prefix for the filename (e.g., 'aws_gpu_prices', 'azure_gpu_prices')

    Returns:
        Path to saved CSV file as string
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.utcnow().strftime('%Y-%m-%d')
    filename = f'{filename_prefix}_{date_str}.csv'
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    return str(filepath)
