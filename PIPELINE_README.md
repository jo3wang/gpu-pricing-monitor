# Multi-Cloud GPU/TPU Pricing Monitor

A production-ready pipeline for monitoring GPU and TPU pricing across AWS, Azure, and GCP with automated data cleaning, outlier removal, and interactive dashboards.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           DATA COLLECTORS               │
│  (AWS, Azure, GCP GPUs + TPUs)          │
└─────────────────────────────────────────┘
           |           |           |
           v           v           v
┌─────────────────────────────────────────┐
│            RAW DAILY CSVs               │
│                 (data/)                 │
└─────────────────────────────────────────┘
                     |
                     v
┌─────────────────────────────────────────┐
│         UNIFIED ANALYSIS PIPELINE      │
│         (analyze_gpu_prices.py)         │
└─────────────────────────────────────────┘
                     |
                     v
┌─────────────────────────────────────────┐
│   CLEANED, AGGREGATED TIME SERIES      │
│      gpu_pricing_summary.csv           │
└─────────────────────────────────────────┘
                     |
                     v
┌─────────────────────────────────────────┐
│            STREAMLIT DASHBOARD          │
│             (dashboard.py)              │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Pricing Data
```bash
# Run data collectors (can be run in parallel)
python fetch_aws_gpu.py      # AWS GPU pricing
python fetch_azure_gpu.py    # Azure GPU pricing  
python fetch_gcp_accel.py    # GCP GPU + TPU pricing (requires auth)
```

### 3. Process and Clean Data
```bash
# Run unified analysis pipeline
python analyze_gpu_prices.py
```

### 4. Launch Dashboard
```bash
# Start interactive dashboard
streamlit run dashboard.py
```

## 📊 Data Pipeline Features

### Data Collection
- **AWS**: Multi-region GPU pricing with caching (400MB+ pricing files)
- **Azure**: Global GPU pricing via Retail Prices API (45+ regions)
- **GCP**: GPU and TPU pricing via Cloud Billing API

### Data Processing
- **Standardized Schema**: Unified columns across all clouds
- **Data Cleaning**: Removes invalid prices (≤0, NaN, infinite)
- **Outlier Removal**: IQR filtering per accelerator/region/date group
- **Dual Aggregation**: Regional and global views with percentiles

### Supported Accelerators
- **GPUs**: H100, A100, V100, T4, A10G, L4, P100, K80, M60
- **TPUs**: TPUv2, TPUv3, TPUv4, TPUv5e, TPUv5p

## 📁 Output Files

### Raw Data
```
data/aws_gpu_prices_YYYY-MM-DD.csv      # AWS pricing
data/azure_gpu_prices_YYYY-MM-DD.csv    # Azure pricing  
data/gcp_accel_prices_YYYY-MM-DD.csv    # GCP pricing
```

### Processed Data
```
data/gpu_pricing_summary.csv            # Unified time-series summary
```

## 🤖 Daily Automation

```bash
#!/bin/bash
# Daily pricing collection
python fetch_aws_gpu.py
python fetch_azure_gpu.py  
python fetch_gcp_accel.py
python analyze_gpu_prices.py
```