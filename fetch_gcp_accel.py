#!/usr/bin/env python3
"""
GCP GPU and TPU Pricing Monitor
Fetches on-demand pricing for GPUs and TPUs using Google Cloud Billing Catalog API.
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import re

# --- Reliability Improvement: Structured Logging ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Import shared utilities
from utils import validate_df, save_to_csv


# Step 1C: Comprehensive GCP GPU and TPU mapping for standardized schema
GCP_GPU_TPU_MAP = {
    # NVIDIA GPUs
    'nvidia-tesla-k80': ('K80', 1),
    'nvidia-tesla-p100': ('P100', 1),
    'nvidia-tesla-p4': ('P4', 1),
    'nvidia-tesla-v100': ('V100', 1),
    'nvidia-tesla-t4': ('T4', 1),
    'nvidia-tesla-a100': ('A100', 1),
    'nvidia-l4': ('L4', 1),
    'nvidia-h100': ('H100', 1),
    'nvidia-h100-80gb': ('H100', 1),
    
    # TPUs - different generations
    'tpu-v2': ('TPUv2', 1),
    'tpu-v3': ('TPUv3', 1), 
    'tpu-v4': ('TPUv4', 1),
    'tpu-v5e': ('TPUv5e', 1),
    'tpu-v5p': ('TPUv5p', 1),
}


def detect_accel_model_and_count(sku_description: str, resource_family: str = "") -> tuple[str | None, str, int]:
    """
    Maps GCP SKU description to accelerator model, type, and count.
    
    Args:
        sku_description: GCP SKU description
        resource_family: Resource family for context
        
    Returns:
        Tuple of (accelerator_model, accelerator_type, accelerator_count) or (None, None, 0)
    """
    desc_lower = sku_description.lower()
    family_lower = resource_family.lower()
    
    # Check for TPUs first
    if 'tpu' in desc_lower or 'tpu' in family_lower:
        for pattern, (model, count) in GCP_GPU_TPU_MAP.items():
            if pattern.startswith('tpu-') and pattern.replace('tpu-', '') in desc_lower:
                return (model, 'TPU', count)
        
        # Generic TPU fallback
        if 'v2' in desc_lower:
            return ('TPUv2', 'TPU', 1)
        elif 'v3' in desc_lower:
            return ('TPUv3', 'TPU', 1)
        elif 'v4' in desc_lower:
            return ('TPUv4', 'TPU', 1)
        elif 'v5e' in desc_lower:
            return ('TPUv5e', 'TPU', 1)
        elif 'v5p' in desc_lower:
            return ('TPUv5p', 'TPU', 1)
    
    # Check for NVIDIA GPUs
    if 'nvidia' in desc_lower or 'gpu' in desc_lower:
        for pattern, (model, count) in GCP_GPU_TPU_MAP.items():
            if pattern.startswith('nvidia-'):
                clean_pattern = pattern.replace('nvidia-', '').replace('-', ' ')
                if clean_pattern in desc_lower:
                    return (model, 'GPU', count)
        
        # Additional GPU patterns
        if 'h100' in desc_lower:
            return ('H100', 'GPU', 1)
        elif 'a100' in desc_lower:
            return ('A100', 'GPU', 1)
        elif 'v100' in desc_lower:
            return ('V100', 'GPU', 1)
        elif 't4' in desc_lower:
            return ('T4', 'GPU', 1)
        elif 'l4' in desc_lower:
            return ('L4', 'GPU', 1)
        elif 'p100' in desc_lower:
            return ('P100', 'GPU', 1)
        elif 'k80' in desc_lower:
            return ('K80', 'GPU', 1)
    
    return (None, None, 0)


def fetch_gcp_billing_data() -> List[Dict]:
    """
    Fetches GCP pricing data using the Cloud Billing Catalog API (public).
    
    Returns:
        List of pricing SKUs from GCP API
    """
    base_url = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"
    
    all_skus = []
    next_page_token = None
    page_count = 0
    
    logging.info("Fetching GCP pricing data from Cloud Billing Catalog API")
    
    while page_count < 20:  # Limit to avoid infinite loops
        params = {}
        if next_page_token:
            params['pageToken'] = next_page_token
        
        try:
            logging.info(f"Fetching page {page_count + 1}")
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            
            data = resp.json()
            skus = data.get('skus', [])
            all_skus.extend(skus)
            
            next_page_token = data.get('nextPageToken')
            page_count += 1
            
            logging.info(f"Retrieved {len(skus)} SKUs (total: {len(all_skus)})")
            
            if not next_page_token:
                break
                
        except requests.RequestException as e:
            logging.error(f"Error fetching GCP pricing data: {e}")
            break
    
    logging.info(f"Total SKUs retrieved: {len(all_skus)} across {page_count} pages")
    return all_skus


def extract_gcp_accel_prices(pricing_data: List[Dict]) -> pd.DataFrame:
    """
    Extracts GPU and TPU prices from GCP pricing data.
    
    Args:
        pricing_data: List of SKU data from GCP Billing API
        
    Returns:
        DataFrame with GPU/TPU pricing information
    """
    accel_instances = []
    
    for sku in pricing_data:
        description = sku.get('description', '')
        sku_id = sku.get('skuId', '')
        
        # Get category and resource info
        category = sku.get('category', {})
        resource_family = category.get('resourceFamily', '')
        resource_group = category.get('resourceGroup', '')
        
        # Check if this is a GPU or TPU resource
        accelerator_model, accelerator_type, accelerator_count = detect_accel_model_and_count(
            description, resource_family
        )
        
        if accelerator_model is None:
            continue
        
        # Extract pricing information
        pricing_info = sku.get('pricingInfo', [])
        if not pricing_info:
            continue
        
        # Get the primary pricing tier (usually the first one)
        pricing = pricing_info[0]
        pricing_expression = pricing.get('pricingExpression', {})
        
        # Extract tiered rates
        tiered_rates = pricing_expression.get('tieredRates', [])
        if not tiered_rates:
            continue
        
        # Get the base rate (first tier)
        base_rate = tiered_rates[0]
        unit_price = base_rate.get('unitPrice', {})
        
        # Extract price in USD
        currency_code = unit_price.get('currencyCode', '')
        if currency_code != 'USD':
            continue
            
        # Price is in micro-units (1/1,000,000 of the currency)
        units = int(unit_price.get('units', 0))
        nanos = int(unit_price.get('nanos', 0))
        price_usd = units + (nanos / 1_000_000_000)
        
        if price_usd <= 0:
            continue
        
        # Extract usage unit (per hour, per core-hour, etc.)
        usage_unit = pricing_expression.get('usageUnit', '')
        display_quantity = pricing_expression.get('displayQuantity', 1)
        
        # Convert to hourly pricing if needed
        if 'hour' in usage_unit.lower():
            price_per_hour = price_usd * display_quantity
        else:
            # Skip non-hourly pricing for now
            continue
        
        # Get service regions
        service_regions = sku.get('serviceRegions', ['global'])
        
        # Step 1C: Create rows for each region with standardized schema
        for region in service_regions:
            accel_instances.append({
                'timestamp_utc': datetime.utcnow().isoformat(),
                'cloud': 'gcp',
                'region': region.lower() if region != 'global' else 'global',
                'sku_id': sku_id,
                'machine_type': description,
                'accelerator_type': accelerator_type,
                'accelerator_model': accelerator_model,
                'accelerator_count': accelerator_count,
                'price_hour_ondemand': price_per_hour,
                'price_per_accel_hour': price_per_hour / accelerator_count if accelerator_count > 0 else price_per_hour,
                'usage_unit': usage_unit,
                'resource_family': resource_family
            })
    
    return pd.DataFrame(accel_instances)


def main():
    """Main function to fetch and save GCP GPU/TPU pricing."""
    
    # Output directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # --- Reliability Improvement: Logging ---
    logging.info("Fetching GCP GPU/TPU pricing from Cloud Billing Catalog API")
    
    try:
        # Step 1C: Fetch pricing data for compute accelerators
        pricing_data = fetch_gcp_billing_data()
        
        if not pricing_data:
            logging.warning("No pricing data retrieved from GCP Billing API")
            return
        
        # Extract GPU and TPU prices
        logging.info("Extracting GPU and TPU prices from GCP data")
        df = extract_gcp_accel_prices(pricing_data)
        
        if df.empty:
            logging.warning("No GPU/TPU instances found in pricing data")
            return
        
        # --- Reliability Improvement: Logging raw extraction ---
        logging.info(f"Extracted {len(df)} raw GPU/TPU rows from GCP")
        
        # --- Reliability Improvement: Validation ---
        df = validate_df(df, "gcp")

        # Save to CSV
        output_path = save_to_csv(df, data_dir, 'gcp_accel_prices')
        
        # --- Reliability Improvement: Summary logging ---
        logging.info(f"Extracted {len(df)} GPU/TPU rows from GCP")
        logging.info(f"Saved cleaned daily CSV → {output_path}")
        
        # Summary by accelerator type
        type_summary = df.groupby('accelerator_type').agg({
            'accelerator_model': lambda x: ', '.join(sorted(x.unique())),
            'region': 'nunique',
            'price_per_accel_hour': lambda x: f"${x.min():.3f}-${x.max():.3f}" if len(x) > 0 else "No pricing"
        }).rename(columns={'accelerator_model': 'models', 'region': 'regions', 'price_per_accel_hour': 'price_range'})
        
        logging.info("GCP accelerator breakdown:")
        for accel_type, data in type_summary.iterrows():
            logging.info(f"  {accel_type}: {data['models']}, {data['regions']} regions, price range: {data['price_range']}")
        
        unique_models = sorted(df['accelerator_model'].unique())
        logging.info(f"GCP accelerator models found: {unique_models}")
        
    except Exception as e:
        logging.error(f"Error processing GCP pricing data: {e}")
        raise


if __name__ == '__main__':
    main()