#!/usr/bin/env python3
"""
GPU and TPU Pricing Analysis Script
Loads and analyzes pricing data from AWS, Azure, and GCP with data cleaning and aggregation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import seaborn as sns
from typing import Dict, List

# --- Reliability Improvement: Structured Logging ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_all_pricing_data(data_dir: Path) -> pd.DataFrame:
    """
    Step 2A: Load and unify all raw CSVs from AWS, Azure, and GCP.
    Maps to common schema preserving cloud and region information.
    """
    
    all_data = []
    cloud_counts = {}
    
    # Load AWS on-demand data
    aws_files = list(data_dir.glob("aws_gpu_prices_*.csv"))
    aws_rows = 0
    for file_path in aws_files:
        try:
            df = pd.read_csv(file_path)
            # Ensure AWS data has standardized columns
            if 'cloud' not in df.columns:
                df['cloud'] = 'aws'
            if 'accelerator_type' not in df.columns and 'gpu_model' in df.columns:
                df['accelerator_type'] = 'GPU'
                df['accelerator_model'] = df['gpu_model']
                df['accelerator_count'] = df['num_gpus']
                df['price_per_accel_hour'] = df['price_per_gpu_hour']
            
            all_data.append(df)
            aws_rows += len(df)
            logging.info(f"Loaded AWS data: {len(df)} rows from {file_path.name}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
    cloud_counts['aws'] = aws_rows
    
    # Load Azure data
    azure_files = list(data_dir.glob("azure_gpu_prices_*.csv"))
    azure_rows = 0
    for file_path in azure_files:
        try:
            df = pd.read_csv(file_path)
            # Ensure Azure data has standardized columns
            if 'cloud' not in df.columns:
                df['cloud'] = 'azure'
            if 'accelerator_type' not in df.columns and 'gpu_model' in df.columns:
                df['accelerator_type'] = 'GPU'
                df['accelerator_model'] = df['gpu_model']
                df['accelerator_count'] = df['num_gpus']
                df['price_per_accel_hour'] = df['price_per_gpu_hour']
            
            all_data.append(df)
            azure_rows += len(df)
            logging.info(f"Loaded Azure data: {len(df)} rows from {file_path.name}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
    cloud_counts['azure'] = azure_rows
    
    # Load GCP data
    gcp_files = list(data_dir.glob("gcp_accel_prices_*.csv"))
    gcp_rows = 0
    for file_path in gcp_files:
        try:
            df = pd.read_csv(file_path)
            # GCP data should already have standardized columns
            if 'cloud' not in df.columns:
                df['cloud'] = 'gcp'
            
            all_data.append(df)
            gcp_rows += len(df)
            logging.info(f"Loaded GCP data: {len(df)} rows from {file_path.name}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
    cloud_counts['gcp'] = gcp_rows
    
    if not all_data:
        logging.error("No pricing data files found!")
        return pd.DataFrame()
    
    # --- Reliability Improvement: Loading summary ---
    total_loaded = sum(cloud_counts.values())
    logging.info(f"Loaded {cloud_counts['aws']} AWS rows, {cloud_counts['azure']} Azure rows, {cloud_counts['gcp']} GCP rows")
    
    # Step 2A: Combine all data into unified DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['timestamp_utc'] = pd.to_datetime(combined_df['timestamp_utc'])
    combined_df['date'] = combined_df['timestamp_utc'].dt.date
    
    # --- Reliability Improvement: Schema validation ---
    required_columns = [
        'timestamp_utc', 'cloud', 'region', 'accelerator_type', 
        'accelerator_model', 'accelerator_count', 'price_per_accel_hour'
    ]
    
    missing_cols = [col for col in required_columns if col not in combined_df.columns]
    if missing_cols:
        logging.error(f"Missing required columns after concatenation: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Check for unknown columns
    known_cols = set(required_columns + ['date', 'timestamp_utc', 'instance_type', 'vm_sku', 'sku_id', 
                     'machine_type', 'price_hour_ondemand', 'price_hour_payg', 'usage_unit', 'resource_family'])
    unknown_cols = set(combined_df.columns) - known_cols
    if unknown_cols:
        logging.warning(f"Unknown columns detected: {list(unknown_cols)}")
    
    logging.info(f"Total combined data: {len(combined_df)} rows")
    logging.info(f"Clouds: {', '.join(sorted(combined_df['cloud'].unique()))}")
    logging.info(f"Accelerator types: {', '.join(sorted(combined_df['accelerator_type'].unique()))}")
    
    return combined_df


# --- AWS Spot GPU integration ---
def load_spot_pricing_data(data_dir: Path) -> pd.DataFrame:
    """
    Load AWS spot pricing data from CSV files.
    
    Args:
        data_dir: Directory containing spot pricing CSV files
        
    Returns:
        DataFrame with spot pricing data
    """
    logging.info("Loading AWS Spot pricing data")
    
    # Load AWS spot data
    spot_files = list(data_dir.glob("aws_gpu_spot_prices_*.csv"))
    if not spot_files:
        logging.warning("No AWS spot pricing files found")
        return pd.DataFrame()
    
    spot_data = []
    spot_rows = 0
    
    for file_path in spot_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                spot_data.append(df)
                spot_rows += len(df)
                logging.info(f"Loaded AWS spot data: {len(df)} rows from {file_path.name}")
        except Exception as e:
            logging.error(f"Error loading spot data from {file_path}: {e}")
    
    if not spot_data:
        logging.warning("No valid spot pricing data loaded")
        return pd.DataFrame()
    
    # Combine spot data
    combined_spot = pd.concat(spot_data, ignore_index=True)
    combined_spot['timestamp_utc'] = pd.to_datetime(combined_spot['timestamp_utc'])
    combined_spot['date'] = combined_spot['timestamp_utc'].dt.date
    
    logging.info(f"Total AWS spot rows loaded: {spot_rows}")
    return combined_spot


def clean_pricing_data(df: pd.DataFrame, is_spot_data: bool = False) -> pd.DataFrame:
    """
    Step 3: Data cleaning and outlier removal using IQR filtering.
    
    Args:
        df: DataFrame to clean
        is_spot_data: Whether this is spot pricing data (uses different column names)
    """
    logging.info("Starting data cleaning and outlier removal")
    
    initial_rows = len(df)
    logging.info(f"Initial rows: {initial_rows:,}")
    
    # Determine price column name based on data type
    price_col = 'price_per_accel_hour_spot' if is_spot_data else 'price_per_accel_hour'
    
    # Check if the expected price column exists
    if price_col not in df.columns:
        if is_spot_data and 'price_per_accel_hour' in df.columns:
            price_col = 'price_per_accel_hour'
        else:
            raise ValueError(f"Expected price column '{price_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # --- Reliability Improvement: Anomaly detection ---
    high_price_threshold = 200.0  # $200/hour per accelerator
    
    # Check for unusually high prices
    high_prices = df[df[price_col] > high_price_threshold]
    if len(high_prices) > 0:
        for _, row in high_prices.iterrows():
            logging.warning(f"Detected unusually high price: ${row[price_col]:.2f}/hour for {row['cloud']} {row['accelerator_model']} in {row['region']}")
    
    # Check for new accelerator models
    existing_models = set()
    for _, row in df.iterrows():
        model = row['accelerator_model']
        if model not in existing_models:
            logging.info(f"New accelerator model discovered: {model}")
            existing_models.add(model)
    
    # Step 3: Drop invalid rows
    logging.info("Removing invalid rows...")
    
    # Remove rows with invalid pricing
    invalid_price = (df[price_col].isna()) | \
                   (df[price_col] <= 0) | \
                   (~np.isfinite(df[price_col]))
    
    df_clean = df[~invalid_price].copy()
    removed_invalid = initial_rows - len(df_clean)
    logging.info(f"Removed {removed_invalid:,} rows with invalid pricing")
    
    # Step 3: IQR outlier filtering per group
    logging.info("Applying IQR outlier filtering...")
    
    # Group by key dimensions for outlier detection
    grouping_cols = ['date', 'cloud', 'accelerator_type', 'accelerator_model', 'region']
    outlier_removed = 0
    
    cleaned_groups = []
    for name, group in df_clean.groupby(grouping_cols):
        if len(group) < 4:
            # Skip IQR filtering for small groups
            cleaned_groups.append(group)
            continue
        
        # Compute IQR for the price column
        Q1 = group[price_col].quantile(0.25)
        Q3 = group[price_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # IQR bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter outliers
        outlier_mask = (group[price_col] >= lower_bound) & \
                      (group[price_col] <= upper_bound)
        
        outliers_in_group = len(group) - outlier_mask.sum()
        outlier_removed += outliers_in_group
        
        if outliers_in_group > 0:
            logging.debug(f"Removed {outliers_in_group} outliers from {name}")
        
        cleaned_groups.append(group[outlier_mask])
    
    df_final = pd.concat(cleaned_groups, ignore_index=True)
    final_rows = len(df_final)
    
    logging.info(f"Total outliers removed: {outlier_removed:,}")
    logging.info(f"Cleaning summary: {initial_rows:,} → {final_rows:,} rows ({final_rows/initial_rows*100:.1f}% retained)")
    
    return df_final


def aggregate_spot_pricing_data(df_spot: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4B: AWS Spot pricing aggregation for spot data.
    
    Args:
        df_spot: Cleaned spot pricing DataFrame
        
    Returns:
        Aggregated spot pricing DataFrame
    """
    logging.info("Starting spot pricing aggregation")
    
    if df_spot.empty:
        logging.warning("Empty spot pricing data provided")
        return pd.DataFrame()
    
    aggregation_results = []
    
    # Spot regional aggregation
    logging.info("Performing spot regional aggregation...")
    regional_grouping = ['date', 'cloud', 'region', 'accelerator_type', 'accelerator_model']
    
    regional_agg = df_spot.groupby(regional_grouping)['price_per_accel_hour_spot'].agg([
        ('num_instances_spot', 'count'),
        ('median_price_per_accel_hour_spot', 'median'),
        ('p10_price_per_accel_hour_spot', lambda x: x.quantile(0.1)),
        ('p90_price_per_accel_hour_spot', lambda x: x.quantile(0.9))
    ]).round(4).reset_index()
    
    logging.info(f"Spot regional aggregations: {len(regional_agg):,} records")
    aggregation_results.append(regional_agg)
    
    # Spot global aggregation
    logging.info("Performing spot global aggregation...")
    global_grouping = ['date', 'cloud', 'accelerator_type', 'accelerator_model']
    
    global_agg = df_spot.groupby(global_grouping)['price_per_accel_hour_spot'].agg([
        ('num_instances_spot', 'count'),
        ('median_price_per_accel_hour_spot', 'median'),
        ('p10_price_per_accel_hour_spot', lambda x: x.quantile(0.1)),
        ('p90_price_per_accel_hour_spot', lambda x: x.quantile(0.9))
    ]).round(4).reset_index()
    
    # Set region to "ALL" for global view
    global_agg['region'] = 'ALL'
    
    # Reorder columns to match regional view
    global_agg = global_agg[regional_agg.columns]
    
    logging.info(f"Spot global aggregations: {len(global_agg):,} records")
    aggregation_results.append(global_agg)
    
    # Combine both views
    spot_summary_df = pd.concat(aggregation_results, ignore_index=True)
    
    logging.info(f"Total spot summary records: {len(spot_summary_df):,}")
    
    return spot_summary_df


def aggregate_pricing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Regional and Global aggregations for on-demand pricing summary CSV.
    """
    logging.info("Starting on-demand pricing aggregation")
    
    aggregation_results = []
    
    # Step 4: Regional view aggregation
    logging.info("Performing on-demand regional view aggregation...")
    regional_grouping = ['date', 'cloud', 'region', 'accelerator_type', 'accelerator_model']
    
    regional_agg = df.groupby(regional_grouping)['price_per_accel_hour'].agg([
        ('num_instances', 'count'),
        ('median_price_per_accel_hour_ondemand', 'median'),
        ('p10_price_per_accel_hour_ondemand', lambda x: x.quantile(0.1)),
        ('p90_price_per_accel_hour_ondemand', lambda x: x.quantile(0.9))
    ]).round(4).reset_index()
    
    logging.info(f"On-demand regional aggregations: {len(regional_agg):,} records")
    aggregation_results.append(regional_agg)
    
    # Step 4: Global view (per cloud) aggregation
    logging.info("Performing on-demand global view aggregation...")
    global_grouping = ['date', 'cloud', 'accelerator_type', 'accelerator_model']
    
    global_agg = df.groupby(global_grouping)['price_per_accel_hour'].agg([
        ('num_instances', 'count'),
        ('median_price_per_accel_hour_ondemand', 'median'),
        ('p10_price_per_accel_hour_ondemand', lambda x: x.quantile(0.1)),
        ('p90_price_per_accel_hour_ondemand', lambda x: x.quantile(0.9))
    ]).round(4).reset_index()
    
    # Set region to "ALL" for global view
    global_agg['region'] = 'ALL'
    
    # Reorder columns to match regional view
    global_agg = global_agg[regional_agg.columns]
    
    logging.info(f"On-demand global aggregations: {len(global_agg):,} records")
    aggregation_results.append(global_agg)
    
    # Combine both views
    summary_df = pd.concat(aggregation_results, ignore_index=True)
    
    logging.info(f"Total on-demand summary records: {len(summary_df):,}")
    
    # Summary by accelerator type
    summary_by_type = summary_df.groupby(['accelerator_type', 'cloud']).size().reset_index(name='records')
    logging.info("On-demand summary by accelerator type:")
    for _, row in summary_by_type.iterrows():
        logging.info(f"  {row['accelerator_type']} ({row['cloud']}): {row['records']} records")
    
    return summary_df


def merge_spot_and_ondemand_data(df_ondemand: pd.DataFrame, df_spot: pd.DataFrame) -> pd.DataFrame:
    """
    Merges on-demand and spot pricing data and calculates spot discounts.
    
    Args:
        df_ondemand: On-demand pricing aggregation DataFrame  
        df_spot: Spot pricing aggregation DataFrame
        
    Returns:
        Merged DataFrame with spot discount calculations
    """
    logging.info("Merging spot and on-demand pricing data")
    
    if df_spot.empty:
        logging.warning("No spot data to merge - returning on-demand data only")
        # Add empty spot columns for consistency
        df_ondemand['median_price_per_accel_hour_spot'] = np.nan
        df_ondemand['spot_discount_median'] = np.nan
        df_ondemand['spot_spread_abs'] = np.nan
        df_ondemand['spot_spread_pct'] = np.nan
        return df_ondemand
    
    # Merge on key dimensions
    merge_cols = ['date', 'cloud', 'region', 'accelerator_type', 'accelerator_model']
    
    merged_df = df_ondemand.merge(
        df_spot[merge_cols + ['median_price_per_accel_hour_spot']],
        on=merge_cols,
        how='left'
    )
    
    # Calculate spot discount metrics
    merged_df['spot_discount_median'] = np.where(
        (merged_df['median_price_per_accel_hour_ondemand'] > 0) & 
        (merged_df['median_price_per_accel_hour_spot'].notna()),
        1 - (merged_df['median_price_per_accel_hour_spot'] / merged_df['median_price_per_accel_hour_ondemand']),
        np.nan
    )
    
    # Absolute spread (on-demand - spot)
    merged_df['spot_spread_abs'] = np.where(
        merged_df['median_price_per_accel_hour_spot'].notna(),
        merged_df['median_price_per_accel_hour_ondemand'] - merged_df['median_price_per_accel_hour_spot'],
        np.nan
    )
    
    # Percentage spread  
    merged_df['spot_spread_pct'] = merged_df['spot_discount_median'] * 100
    
    # Round discount calculations
    merged_df['spot_discount_median'] = merged_df['spot_discount_median'].round(4)
    merged_df['spot_spread_abs'] = merged_df['spot_spread_abs'].round(4)
    merged_df['spot_spread_pct'] = merged_df['spot_spread_pct'].round(2)
    
    # Count records with spot data
    spot_records = merged_df['median_price_per_accel_hour_spot'].notna().sum()
    total_records = len(merged_df)
    
    logging.info(f"Merged {spot_records}/{total_records} records have spot pricing data")
    
    if spot_records > 0:
        avg_discount = merged_df['spot_discount_median'].mean()
        logging.info(f"Average spot discount: {avg_discount:.1%}")
        
        # Top discounts for logging
        top_discounts = merged_df[merged_df['spot_discount_median'].notna()].nlargest(3, 'spot_discount_median')
        logging.info("Top spot discounts:")
        for _, row in top_discounts.iterrows():
            logging.info(f"  {row['date']} {row['cloud']} {row['accelerator_model']} "
                        f"({row['region']}): {row['spot_discount_median']:.1%} discount")
    
    return merged_df


# --- Stress Index: computation functions ---
def compute_stress_index(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Computes GPU Stress Index for global rows (region == 'ALL') based on price movements.
    
    Args:
        df_summary: Summary DataFrame with pricing data
        
    Returns:
        DataFrame with stress_core column added for global rows
    """
    logging.info("Computing GPU Stress Index for global pricing series")
    
    # Filter to global rows only and sort by date
    global_rows = df_summary[df_summary['region'] == 'ALL'].copy()
    if global_rows.empty:
        logging.warning("No global rows found for stress index computation")
        return df_summary
    
    global_rows['date'] = pd.to_datetime(global_rows['date'])
    global_rows = global_rows.sort_values(['cloud', 'accelerator_model', 'date'])
    
    stress_results = []
    series_count = 0
    insufficient_history = 0
    
    # Process each (cloud, accelerator_model) series
    for (cloud, model), series_df in global_rows.groupby(['cloud', 'accelerator_model']):
        series_df = series_df.sort_values('date').reset_index(drop=True)
        
        # Use on-demand pricing for stress index computation (spot prices are too volatile)
        prices = series_df['median_price_per_accel_hour_ondemand'].values
        
        if len(series_df) < 30:
            insufficient_history += 1
            logging.debug(f"Insufficient history for {cloud} {model}: {len(series_df)} days")
        
        # --- Stress Index: derived metrics (roc_30, dd_365, accel_30) ---
        logging.debug(f"Computing stress metrics for {cloud} {model} ({len(series_df)} data points)")
        
        # Initialize metric arrays
        roc_30 = np.full(len(prices), np.nan)
        dd_365 = np.full(len(prices), np.nan)  
        accel_30 = np.full(len(prices), np.nan)
        
        # a) 30-day rate of change: roc_30(t) = (P_t - P_{t-30}) / P_{t-30}
        for i in range(30, len(prices)):
            if prices[i-30] > 0:  # Avoid division by zero
                roc_30[i] = (prices[i] - prices[i-30]) / prices[i-30]
        
        # b) 365-day drawdown from peak: dd_365(t) = (P_t - P_max(t)) / P_max(t)
        for i in range(len(prices)):
            # Rolling 365-day max (or all available history if less)
            start_idx = max(0, i - 365 + 1)
            p_max = np.max(prices[start_idx:i+1])
            if p_max > 0:
                dd_365[i] = (prices[i] - p_max) / p_max
        
        # c) 30-day acceleration: accel_30(t) = (P_t - 2*P_{t-30} + P_{t-60}) / 30
        for i in range(60, len(prices)):
            accel_30[i] = (prices[i] - 2*prices[i-30] + prices[i-60]) / 30.0
        
        # --- Stress Index: z-score normalization ---
        logging.debug(f"Computing z-scores for {cloud} {model}")
        
        # Convert to pandas Series for rolling operations
        series_df_work = series_df.copy()
        series_df_work['roc_30'] = roc_30
        series_df_work['dd_365'] = dd_365
        series_df_work['accel_30'] = accel_30
        
        # Compute rolling z-scores (180-day window, minimum 30 periods)
        rolling_window = min(180, len(series_df_work))
        min_periods = min(30, len(series_df_work) // 2)
        
        # Z-score for roc_30
        roc_mean = series_df_work['roc_30'].rolling(window=rolling_window, min_periods=min_periods).mean()
        roc_std = series_df_work['roc_30'].rolling(window=rolling_window, min_periods=min_periods).std()
        series_df_work['z_roc_30'] = np.where(
            (roc_std > 0) & (~roc_std.isna()) & (~series_df_work['roc_30'].isna()),
            (series_df_work['roc_30'] - roc_mean) / roc_std,
            0.0
        )
        
        # Z-score for dd_365  
        dd_mean = series_df_work['dd_365'].rolling(window=rolling_window, min_periods=min_periods).mean()
        dd_std = series_df_work['dd_365'].rolling(window=rolling_window, min_periods=min_periods).std()
        series_df_work['z_dd_365'] = np.where(
            (dd_std > 0) & (~dd_std.isna()) & (~series_df_work['dd_365'].isna()),
            (series_df_work['dd_365'] - dd_mean) / dd_std,
            0.0
        )
        
        # Z-score for accel_30
        accel_mean = series_df_work['accel_30'].rolling(window=rolling_window, min_periods=min_periods).mean()
        accel_std = series_df_work['accel_30'].rolling(window=rolling_window, min_periods=min_periods).std()
        series_df_work['z_accel_30'] = np.where(
            (accel_std > 0) & (~accel_std.isna()) & (~series_df_work['accel_30'].isna()),
            (series_df_work['accel_30'] - accel_mean) / accel_std,
            0.0
        )
        
        # --- Stress Index: core index computation (stress_core) ---
        # stress_core = 0.4 * (-z_roc_30) + 0.4 * (-z_dd_365) + 0.2 * (-z_accel_30)
        # Higher stress_core = more downside stress
        
        def compute_stress_core_row(row):
            z_vals = [row['z_roc_30'], row['z_dd_365'], row['z_accel_30']]
            weights = [0.4, 0.4, 0.2]
            
            # Check for valid (non-NaN) values
            valid_pairs = [(z, w) for z, w in zip(z_vals, weights) if not pd.isna(z)]
            
            if not valid_pairs:
                return np.nan
                
            # Compute weighted sum with sign flip (higher = more stress)
            stress = sum(-z * w for z, w in valid_pairs)
            return stress
        
        series_df_work['stress_core'] = series_df_work.apply(compute_stress_core_row, axis=1)
        
        # Store results
        stress_results.append(series_df_work[['date', 'cloud', 'region', 'accelerator_type', 
                                           'accelerator_model', 'stress_core']])
        series_count += 1
    
    if not stress_results:
        logging.warning("No stress index computed for any series")
        return df_summary
    
    # Combine all stress results
    stress_df = pd.concat(stress_results, ignore_index=True)
    
    # Ensure consistent date format for merging
    stress_df['date'] = stress_df['date'].dt.date
    
    # Merge stress_core back into the summary DataFrame
    df_summary_with_stress = df_summary.merge(
        stress_df[['date', 'cloud', 'accelerator_model', 'stress_core']],
        on=['date', 'cloud', 'accelerator_model'],
        how='left'
    )
    
    # --- Stress Index: logging and sanity checks ---
    valid_stress_count = df_summary_with_stress['stress_core'].notna().sum()
    logging.info(f"Computed stress_core for {series_count} (cloud, accelerator_model) series")
    logging.info(f"Total rows with non-NaN stress_core: {valid_stress_count}")
    
    if insufficient_history > 0:
        logging.info(f"{insufficient_history} series had insufficient history for full metrics")
    
    # Show top stress values
    top_stress = df_summary_with_stress[
        (df_summary_with_stress['stress_core'].notna()) &
        (df_summary_with_stress['region'] == 'ALL')
    ].nlargest(5, 'stress_core')
    
    if not top_stress.empty:
        logging.info("Top stress_core values:")
        for _, row in top_stress.iterrows():
            logging.info(f"  {row['date']} {row['cloud']} {row['accelerator_model']} stress_core={row['stress_core']:.2f}")
    
    return df_summary_with_stress


def analyze_pricing_by_accelerator_model(df: pd.DataFrame):
    """Analyze pricing breakdown by GPU model."""
    
    print("\n" + "="*60)
    print("GPU PRICING ANALYSIS")
    print("="*60)
    
    # Filter out zero prices for analysis
    non_zero = df[df['price_per_gpu_hour'] > 0].copy()
    
    print(f"\nDataset Overview:")
    print(f"  Total records: {len(df):,}")
    print(f"  Records with pricing: {len(non_zero):,} ({len(non_zero)/len(df)*100:.1f}%)")
    print(f"  Providers: {', '.join(df['provider'].unique())}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    print(f"\nGPU Models Found:")
    gpu_counts = df['gpu_model'].value_counts()
    for gpu, count in gpu_counts.items():
        print(f"  {gpu}: {count:,} instances")
    
    print(f"\nPricing Summary by GPU Model:")
    pricing_summary = non_zero.groupby(['provider', 'gpu_model'])['price_per_gpu_hour'].agg([
        'count', 'min', 'median', 'max', 'std'
    ]).round(3)
    
    print(pricing_summary.to_string())
    
    print(f"\nRegional Coverage:")
    region_summary = df.groupby(['provider', 'region']).size().reset_index(name='instances')
    providers_regions = region_summary.groupby('provider')['region'].count()
    for provider, count in providers_regions.items():
        print(f"  {provider}: {count} regions")


def plot_gpu_pricing_comparison(df: pd.DataFrame, output_dir: Path):
    """Create visualizations of GPU pricing."""
    
    # Filter out zero prices and focus on main GPU models
    non_zero = df[df['price_per_gpu_hour'] > 0].copy()
    main_gpus = ['H100', 'A100', 'V100', 'T4']
    gpu_data = non_zero[non_zero['gpu_model'].isin(main_gpus)]
    
    if gpu_data.empty:
        print("No data available for main GPU models (H100, A100, V100, T4)")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GPU Pricing Analysis - AWS vs Azure', fontsize=16, fontweight='bold')
    
    # 1. Box plot of pricing by GPU model and provider
    if len(gpu_data) > 0:
        sns.boxplot(data=gpu_data, x='gpu_model', y='price_per_gpu_hour', 
                   hue='provider', ax=ax1)
        ax1.set_title('Price Distribution by GPU Model')
        ax1.set_ylabel('Price per GPU Hour (USD)')
        ax1.set_xlabel('GPU Model')
    
    # 2. Regional price comparison for H100 (if available)
    h100_data = gpu_data[gpu_data['gpu_model'] == 'H100']
    if not h100_data.empty:
        regional_h100 = h100_data.groupby(['provider', 'region'])['price_per_gpu_hour'].median().reset_index()
        sns.barplot(data=regional_h100, x='region', y='price_per_gpu_hour', 
                   hue='provider', ax=ax2)
        ax2.set_title('H100 Regional Pricing Comparison')
        ax2.set_ylabel('Median Price per H100 Hour (USD)')
        ax2.set_xlabel('Region')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No H100 pricing data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('H100 Regional Pricing (No Data)')
    
    # 3. Price distribution histogram
    if len(gpu_data) > 0:
        for provider in gpu_data['provider'].unique():
            provider_data = gpu_data[gpu_data['provider'] == provider]
            ax3.hist(provider_data['price_per_gpu_hour'], alpha=0.7, 
                    label=provider, bins=20)
        ax3.set_title('Price Distribution Across All GPU Models')
        ax3.set_xlabel('Price per GPU Hour (USD)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    
    # 4. Instance count by provider and GPU
    instance_counts = gpu_data.groupby(['provider', 'gpu_model']).size().reset_index(name='count')
    sns.barplot(data=instance_counts, x='gpu_model', y='count', 
               hue='provider', ax=ax4)
    ax4.set_title('Instance Count by GPU Model')
    ax4.set_ylabel('Number of Instance Types')
    ax4.set_xlabel('GPU Model')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / f'gpu_pricing_analysis_{pd.Timestamp.now().strftime("%Y-%m-%d")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    
    plt.show()


def main():
    """
    Main analysis function implementing the complete pipeline:
    Load -> Clean -> Aggregate -> Save summary CSV
    """
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    
    if not data_dir.exists():
        logging.error(f"Data directory not found: {data_dir}")
        return
    
    logging.info("="*60)
    logging.info("MULTI-CLOUD GPU/TPU PRICING ANALYSIS PIPELINE")
    logging.info("="*60)
    
    # Step 2A: Load all pricing data
    logging.info("Step 1: Loading pricing data")
    df_raw = load_all_pricing_data(data_dir)
    
    if df_raw.empty:
        logging.error("No pricing data loaded. Make sure to run collectors first:")
        logging.error("  python fetch_aws_gpu.py")
        logging.error("  python fetch_azure_gpu.py") 
        logging.error("  python fetch_gcp_accel.py")
        return
    
    # Step 3: Clean pricing data
    df_clean = clean_pricing_data(df_raw)
    
    if df_clean.empty:
        logging.error("No data remaining after cleaning!")
        return
    
    # Step 4: Aggregate on-demand pricing data
    df_summary_ondemand = aggregate_pricing_data(df_clean)
    
    # Step 4B: Load and aggregate spot pricing data (AWS only)
    logging.info("Step 4B: Loading AWS Spot pricing data")
    df_spot_raw = load_spot_pricing_data(data_dir)
    
    if not df_spot_raw.empty:
        # Clean spot data using the same cleaning function with spot flag
        df_spot_clean = clean_pricing_data(df_spot_raw, is_spot_data=True)
        
        # Aggregate spot data
        df_summary_spot = aggregate_spot_pricing_data(df_spot_clean)
        
        # Merge spot and on-demand data with discount calculations
        df_summary = merge_spot_and_ondemand_data(df_summary_ondemand, df_summary_spot)
        logging.info("Combined on-demand and spot pricing data")
    else:
        logging.warning("No spot pricing data found - using on-demand only")
        # Add empty spot columns for consistency
        df_summary = df_summary_ondemand.copy()
        df_summary['median_price_per_accel_hour_spot'] = np.nan
        df_summary['spot_discount_median'] = np.nan
        df_summary['spot_spread_abs'] = np.nan
        df_summary['spot_spread_pct'] = np.nan
    
    # --- Stress Index: compute stress metrics for global pricing series ---
    logging.info("Step 5: Computing GPU Stress Index")
    df_summary_with_stress = compute_stress_index(df_summary)
    
    # Save the unified time-series summary CSV (now includes stress_core and spot metrics)
    summary_path = data_dir / 'gpu_pricing_summary.csv'
    df_summary_with_stress.to_csv(summary_path, index=False)
    
    # --- Reliability Improvement: Summary log footer ---
    logging.info("="*60)
    logging.info("FINAL DATASET SUMMARY")
    logging.info("="*60)
    logging.info(f"Total cleaned rows: {len(df_clean):,}")
    
    # Cloud breakdown
    cloud_breakdown = df_clean.groupby('cloud').size().to_dict()
    clouds_str = ', '.join([f"{cloud}: {count}" for cloud, count in cloud_breakdown.items()])
    logging.info(f"Clouds: {clouds_str}")
    
    # Accelerator models found
    unique_models = sorted(df_clean['accelerator_model'].unique())
    logging.info(f"Accelerator models: {unique_models}")
    
    logging.info(f"Saved summary to {summary_path}")
    
    # Legacy analysis for comparison
    try:
        analyze_pricing_by_accelerator_model(df_clean)
    except Exception as e:
        logging.error(f"Error in legacy analysis: {e}")
    
    logging.info("="*60)
    logging.info("PIPELINE COMPLETE")
    logging.info("="*60)
    logging.info(f"✓ Processed data from {len(df_raw['cloud'].unique())} clouds")
    logging.info(f"✓ Cleaned and aggregated {len(df_summary_with_stress):,} pricing records")
    logging.info(f"✓ Added stress index for global pricing series")
    logging.info(f"✓ Ready for dashboard: {summary_path}")
    logging.info("To launch dashboard: streamlit run dashboard.py")


if __name__ == '__main__':
    main()