#!/usr/bin/env python3
"""
Azure GPU Pricing Monitor
Fetches pay-as-you-go pricing for GPU VMs and saves to CSV.
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

# --- Reliability Improvement: Structured Logging ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Import shared utilities
from utils import validate_df, save_to_csv


# Step 1B: Comprehensive Azure GPU VM mapping for standardized schema
AZURE_GPU_VM_MAP = {
    # NC-series: NVIDIA Tesla K80, P100, V100, T4, A100
    'nc6s_v3': ("V100", 1),
    'nc12s_v3': ("V100", 2),
    'nc24s_v3': ("V100", 4),
    'nc4as_t4_v3': ("T4", 1),
    'nc8as_t4_v3': ("T4", 2),
    'nc16as_t4_v3': ("T4", 4),
    'nc64as_t4_v3': ("T4", 4),
    'nc24ads_a100_v4': ("A100", 1),
    'nc48ads_a100_v4': ("A100", 2),
    'nc96ads_a100_v4': ("A100", 4),
    
    # ND-series: NVIDIA Tesla P100, V100, A100, H100
    'nd6s_v2': ("P100", 1),
    'nd12s_v2': ("P100", 2),
    'nd24s_v2': ("P100", 4),
    'nd40rs_v2': ("V100", 8),
    'nd96asr_v4': ("A100", 8),
    'nd96amsr_a100_v4': ("A100", 8),
    'nd96isr_h100_v5': ("H100", 8),
    
    # NV-series: NVIDIA Tesla M60, A10
    'nv6': ("M60", 0.5),
    'nv12': ("M60", 1),
    'nv24': ("M60", 2),
    'nv6ads_a10_v5': ("A10", 1),
    'nv12ads_a10_v5': ("A10", 1),
    'nv18ads_a10_v5': ("A10", 1),
    'nv36ads_a10_v5': ("A10", 1),
    'nv36adms_a10_v5': ("A10", 1),
    'nv72ads_a10_v5': ("A10", 2),
}


def detect_accel_model_and_count(vm_sku: str) -> tuple[str | None, int]:
    """
    Maps Azure VM SKU to accelerator model and count using standardized mapping.
    
    Args:
        vm_sku: Azure VM SKU (e.g., 'Standard_NC6s_v3', 'Standard_ND96asr_v4')
        
    Returns:
        Tuple of (accelerator_model, accelerator_count) or (None, 0) if not a GPU VM
    """
    sku_lower = vm_sku.lower().replace('standard_', '')
    
    # Check direct mapping first
    if sku_lower in AZURE_GPU_VM_MAP:
        return AZURE_GPU_VM_MAP[sku_lower]
    
    # Check for partial matches (for spot/low priority variants)
    base_sku = sku_lower.split()[0] if ' ' in sku_lower else sku_lower
    for pattern, (model, count) in AZURE_GPU_VM_MAP.items():
        if pattern in base_sku:
            return (model, count)
    
    # Not a GPU VM
    return (None, 0)


def fetch_azure_pricing_data(api_filters: dict = None) -> List[dict]:
    """
    Fetches Azure retail pricing data using the Azure Retail Prices API.
    
    Args:
        api_filters: Optional filters for the API query
        
    Returns:
        List of pricing items from Azure API
        
    Raises:
        requests.HTTPError: If the HTTP request fails
    """
    base_url = "https://prices.azure.com/api/retail/prices"
    
    # Default filters for compute services
    default_filters = {
        "$filter": "serviceFamily eq 'Compute' and priceType eq 'Consumption'",
        "api-version": "2023-01-01-preview"
    }
    
    if api_filters:
        default_filters.update(api_filters)
    
    all_items = []
    next_page = base_url
    
    logging.info("Fetching Azure pricing data from Retail Prices API")
    page_count = 0
    
    while next_page and page_count < 50:  # Limit pages to avoid infinite loops
        try:
            logging.info(f"Fetching page {page_count + 1}")
            resp = requests.get(next_page, params=default_filters, timeout=30)
            resp.raise_for_status()
            
            data = resp.json()
            items = data.get('Items', [])
            all_items.extend(items)
            
            next_page = data.get('NextPageLink')
            page_count += 1
            
            # Clear filters after first request (they're embedded in NextPageLink)
            default_filters = {}
            
            logging.info(f"Retrieved {len(items)} items (total: {len(all_items)})")
            
        except requests.RequestException as e:
            logging.error(f"Error fetching Azure pricing data: {e}")
            break
    
    logging.info(f"Total items retrieved: {len(all_items)} across {page_count} pages")
    return all_items


def extract_azure_gpu_prices(pricing_data: List[dict]) -> pd.DataFrame:
    """
    Extracts GPU VM prices from Azure pricing data.
    
    Args:
        pricing_data: List of pricing items from Azure API
        
    Returns:
        DataFrame with GPU VM pricing information
    """
    gpu_vms = []
    
    for item in pricing_data:
        product_name = item.get('productName', '')
        sku_name = item.get('skuName', '')
        service_name = item.get('serviceName', '')
        
        # Filter for Virtual Machines service
        if service_name != 'Virtual Machines':
            continue
        
        # Check if this is a GPU VM
        accelerator_model, accelerator_count = detect_accel_model_and_count(sku_name)
        if accelerator_model is None:
            continue
        
        # Extract pricing information
        unit_price = item.get('unitPrice', 0)
        currency_code = item.get('currencyCode', '')
        region = item.get('armRegionName', '')
        location = item.get('location', '')
        
        # Skip if not USD pricing
        if currency_code != 'USD':
            continue
        
        # Skip if no pricing available
        if unit_price is None or unit_price == 0:
            continue
        
        # Step 1B: Use standardized schema with normalized column names
        price_per_accel_hour = unit_price / accelerator_count if accelerator_count > 0 else unit_price
        
        gpu_vms.append({
            'timestamp_utc': datetime.utcnow().isoformat(),
            'cloud': 'azure',
            'region': (location or region).lower().replace(' ', ''),
            'vm_sku': sku_name,
            'accelerator_type': 'GPU',
            'accelerator_model': accelerator_model,
            'accelerator_count': accelerator_count,
            'price_hour_payg': unit_price,
            'price_per_accel_hour': price_per_accel_hour
        })
    
    return pd.DataFrame(gpu_vms)


def main():
    """Main function to fetch and save Azure GPU pricing."""
    
    # Output directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # --- Reliability Improvement: Logging ---
    logging.info("Fetching Azure GPU VM pricing from Retail Prices API")
    
    try:
        # Fetch pricing data for GPU VMs specifically
        gpu_filter = {
            "$filter": (
                "serviceFamily eq 'Compute' and serviceName eq 'Virtual Machines' and "
                "priceType eq 'Consumption' and "
                "(contains(skuName, 'NC') or contains(skuName, 'ND') or contains(skuName, 'NV'))"
            )
        }
        
        pricing_data = fetch_azure_pricing_data(gpu_filter)
        
        if not pricing_data:
            logging.warning("No pricing data retrieved from Azure API")
            return
        
        # Extract GPU prices
        logging.info("Extracting GPU VM prices from Azure data")
        df = extract_azure_gpu_prices(pricing_data)
        
        if df.empty:
            logging.warning("No GPU VMs found in pricing data")
            return
        
        # --- Reliability Improvement: Logging raw extraction ---
        logging.info(f"Extracted {len(df)} raw GPU rows from Azure")
        
        # --- Reliability Improvement: Validation ---
        df = validate_df(df, "azure")

        # Save to CSV
        output_path = save_to_csv(df, data_dir, 'azure_gpu_prices')
        
        # --- Reliability Improvement: Summary logging ---
        logging.info(f"Extracted {len(df)} GPU rows from Azure")
        logging.info(f"Saved cleaned daily CSV → {output_path}")
        
        # Summary by region
        if not df.empty:
            region_summary = df.groupby('region').agg({
                'vm_sku': 'nunique',
                'accelerator_model': lambda x: ', '.join(sorted(x.unique())),
                'price_per_accel_hour': lambda x: f"${x.min():.3f}-${x.max():.3f}" if len(x) > 0 else "No pricing"
            }).rename(columns={'vm_sku': 'unique_skus', 'accelerator_model': 'accelerator_models', 'price_per_accel_hour': 'price_range'})
            
            logging.info("Azure region breakdown:")
            for region, data in region_summary.iterrows():
                logging.info(f"  {region}: {data['unique_skus']} SKUs, models: {data['accelerator_models']}, price range: {data['price_range']}")
        
        unique_models = sorted(df['accelerator_model'].unique())
        logging.info(f"Azure accelerator models found: {unique_models}")
        
    except Exception as e:
        logging.error(f"Error processing Azure pricing data: {e}")
        raise


if __name__ == '__main__':
    main()