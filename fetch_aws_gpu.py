#!/usr/bin/env python3
"""
AWS EC2 GPU Pricing Monitor
Fetches on-demand pricing for GPU instances and saves to CSV.
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Reliability Improvement: Structured Logging ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Import shared utilities
from utils import detect_aws_accel_model_and_count, validate_df, save_to_csv


def fetch_aws_pricing_json(url: str) -> dict:
    """
    Downloads and parses AWS pricing JSON with streaming.
    
    Args:
        url: URL to AWS pricing JSON
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        requests.HTTPError: If the HTTP request fails
        json.JSONDecodeError: If JSON parsing fails
    """
    logging.info(f"Downloading AWS pricing data from {url}")
    logging.warning("This file is ~400MB and may take a few minutes")
    
    # Use streaming to handle large file
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    
    logging.info("Downloading pricing data...")
    content_length = int(resp.headers.get('content-length', 0))
    downloaded = 0
    chunks = []
    
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            if content_length > 0 and downloaded % (10 * 1024 * 1024) == 0:  # Progress every 10MB
                progress = (downloaded / content_length) * 100
                logging.info(f"Downloaded {progress:.1f}% ({downloaded // 1024 // 1024} MB)")
    
    logging.info("Parsing JSON...")
    content = b''.join(chunks)
    return json.loads(content.decode('utf-8'))


def extract_gpu_prices(pricing_data: dict, region: str) -> pd.DataFrame:
    """
    Extracts GPU instance prices from AWS pricing data.
    
    Args:
        pricing_data: Parsed AWS pricing JSON
        region: AWS region name (e.g., 'us-east-1')
        
    Returns:
        DataFrame with GPU pricing information
    """
    products = pricing_data.get('products', {})
    terms = pricing_data.get('terms', {})
    on_demand_terms = terms.get('OnDemand', {})
    
    gpu_instances = []
    
    # Iterate through products to find GPU instances
    for sku, product_info in products.items():
        attributes = product_info.get('attributes', {})
        instance_type = attributes.get('instanceType', '')
        location = attributes.get('location', '')
        tenancy = attributes.get('tenancy', '')
        operating_system = attributes.get('operatingSystem', '')
        
        # Filter: GPU instance types, matching region, shared tenancy, Linux
        accelerator_model, accelerator_count = detect_aws_accel_model_and_count(instance_type)
        if accelerator_model is None:
            continue
        
        if location.lower() != region.lower():
            continue
        
        if tenancy.lower() != 'shared':
            continue
        
        if operating_system.lower() != 'linux':
            continue
        
        # Get on-demand pricing for this SKU
        sku_terms = on_demand_terms.get(sku, {})
        if not sku_terms:
            continue
        
        # Extract price (usually nested under priceDimensions)
        price_hour = None
        for term_code, term_data in sku_terms.items():
            price_dimensions = term_data.get('priceDimensions', {})
            for price_code, price_info in price_dimensions.items():
                price_per_unit = price_info.get('pricePerUnit', {})
                price_hour = price_per_unit.get('USD')
                if price_hour:
                    try:
                        price_hour = float(price_hour)
                        break
                    except (ValueError, TypeError):
                        continue
            if price_hour:
                break
        
        if price_hour is None:
            continue
        
        # Step 1A: Use standardized schema with normalized column names
        price_per_accel_hour = price_hour / accelerator_count if accelerator_count > 0 else price_hour
        
        gpu_instances.append({
            'timestamp_utc': datetime.utcnow().isoformat(),
            'cloud': 'aws',
            'region': region,  # Will be mapped to region code in main()
            'instance_type': instance_type,
            'accelerator_type': 'GPU',
            'accelerator_model': accelerator_model,
            'accelerator_count': accelerator_count,
            'price_hour_ondemand': price_hour,
            'price_per_accel_hour': price_per_accel_hour
        })
    
    return pd.DataFrame(gpu_instances)


def main():
    """Main function to fetch and save AWS GPU pricing."""
    # Step 1A: AWS regions to collect pricing for with region code mapping
    REGIONS = {
        'us-east-1': 'US East (N. Virginia)',
        'us-west-2': 'US West (Oregon)', 
        'eu-west-1': 'EU (Ireland)'
    }
    
    # Reverse mapping for region code lookup
    REGION_NAME_TO_CODE = {name: code for code, name in REGIONS.items()}
    
    # Output directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # --- Reliability Improvement: Logging ---
    logging.info(f"Fetching AWS pricing for {len(REGIONS)} regions: {list(REGIONS.keys())}")
    
    all_gpu_data = []
    
    for region_code, region_name in REGIONS.items():
        logging.info(f"Processing region: {region_name} ({region_code})")
        
        pricing_url = f'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{region_code}/index.json'
        cache_path = data_dir / f"aws_ec2_pricing_{region_code}.json"
        
        try:
            # Try to load from cache first
            if cache_path.exists():
                logging.info(f"Loading pricing data from cache: {cache_path}")
                with cache_path.open("r", encoding="utf-8") as f:
                    pricing_data = json.load(f)
            else:
                # Fetch pricing data and cache it
                pricing_data = fetch_aws_pricing_json(pricing_url)
                logging.info(f"Caching pricing data to: {cache_path}")
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(pricing_data, f)
            
            # Extract GPU prices for this region
            logging.info(f"Extracting GPU instance prices for {region_name}")
            region_df = extract_gpu_prices(pricing_data, region_name)
            
            if region_df.empty:
                logging.warning(f"No GPU instances found for {region_name}")
                continue
            
            # --- Reliability Improvement: Logging raw extraction ---
            logging.info(f"Extracted {len(region_df)} raw GPU rows from {region_name}")
            
            # Step 1A: Map region names back to region codes for standardization
            region_df['region'] = region_df['region'].map(REGION_NAME_TO_CODE).fillna(region_code)
            
            # --- Reliability Improvement: Validation ---
            region_df = validate_df(region_df, "aws")
            
            logging.info(f"Found {len(region_df)} valid GPU instances in {region_name}")
            all_gpu_data.append(region_df)
            
        except requests.RequestException as e:
            logging.error(f"Error fetching pricing data for {region_name}: {e}")
            continue
        except Exception as e:
            logging.error(f"Error processing pricing data for {region_name}: {e}")
            continue
    
    # Combine all regions
    if not all_gpu_data:
        logging.error("No GPU pricing data collected from any region!")
        return
        
    logging.info(f"Combining data from {len(all_gpu_data)} regions")
    df = pd.concat(all_gpu_data, ignore_index=True)
    
    if df.empty:
        logging.warning("No GPU instances found in pricing data")
        return
    
    # --- Reliability Improvement: Final validation ---
    df = validate_df(df, "aws")

    # Save to CSV
    output_path = save_to_csv(df, data_dir, 'aws_gpu_prices')
    
    # --- Reliability Improvement: Summary logging ---
    logging.info(f"Extracted {len(df)} GPU rows from AWS")
    logging.info(f"Saved cleaned daily CSV → {output_path}")
    
    # Summary by region
    region_summary = df.groupby('region').agg({
        'instance_type': 'nunique',
        'accelerator_model': lambda x: ', '.join(sorted(x.unique())),
        'price_per_accel_hour': lambda x: f"${x[x > 0].min():.3f}-${x.max():.3f}" if len(x[x > 0]) > 0 else "No pricing"
    }).rename(columns={'instance_type': 'unique_instances', 'accelerator_model': 'accelerator_models', 'price_per_accel_hour': 'price_range'})
    
    logging.info("AWS region breakdown:")
    for region, data in region_summary.iterrows():
        logging.info(f"  {region}: {data['unique_instances']} instances, models: {data['accelerator_models']}, price range: {data['price_range']}")
    
    unique_models = sorted(df['accelerator_model'].unique())
    logging.info(f"AWS accelerator models found: {unique_models}")
    
    try:
        pass  # Main processing complete
    except Exception as e:
        logging.error(f"Error in main processing: {e}")
        raise


if __name__ == '__main__':
    main()

