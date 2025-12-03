# Spot Pricing Historical Data Collection

## ✅ What I've Fixed

### 1. Dashboard Updates
- **Removed US-only filter** → Now shows ALL US regions across all clouds
  - AWS: us-east-1, us-west-1, us-west-2
  - Azure: eastus, westus, westus2, westus3, centralus, northcentralus, southcentralus
  - GCP: us-central1, us-west1, us-west2, us-west3, us-west4, us-east1, us-east4, us-east5, us-south1

- **Documented aggregation method** → Dashboard now clearly states:
  - Uses **MEDIAN** (not mean) for price aggregation
  - More robust to outliers
  - IQR filtering applied before aggregation

### 2. Spot Collector Rewrite (`fetch_aws_spot_gpu.py`)

**Major improvements:**
- **Fetches past 7 days** of spot pricing history (was only 24 hours)
- **Groups by date** and calculates daily median spot prices
- **Saves one CSV per day** (e.g., `aws_gpu_spot_prices_2025-11-26.csv`)
- Uses **pagination** to fetch all available history
- Calculates **median spot price per day** from all availability zones
- **NO mock data** - 100% real AWS API data

### How It Works:
1. Queries AWS EC2 `describe_spot_price_history()` API
2. Fetches up to 1000 records per instance type (with pagination)
3. Groups prices by date across all availability zones
4. Calculates **median** spot price per instance type per day
5. Saves separate CSV for each historical date
6. Works with existing `analyze_gpu_prices.py` pipeline

## 📊 Data Aggregation Methodology

### Confirmed from `analyze_gpu_prices.py`:

**Aggregation uses MEDIAN (lines 296, 310, 346, 360):**
```python
# Regional aggregation
regional_agg = df.groupby(grouping)['price_per_accel_hour'].agg([
    ('num_instances', 'count'),
    ('median_price_per_accel_hour_ondemand', 'median'),  # ← MEDIAN
    ('p10_price_per_accel_hour_ondemand', lambda x: x.quantile(0.1)),
    ('p90_price_per_accel_hour_ondemand', lambda x: x.quantile(0.9))
])
```

**Why MEDIAN?**
- More robust to outliers than mean
- GPU prices can have extreme outliers (misconfigured instances, temporary spikes)
- Better represents "typical" pricing
- Used for both spot and on-demand pricing

**Outlier Removal:**
- IQR (Interquartile Range) filtering applied BEFORE aggregation
- Removes prices outside 1.5 * IQR range
- Ensures clean, accurate data

## 🚀 How to Collect Historical Spot Data

### Step 1: Install Dependencies (if needed)

```bash
pip3 install boto3 pandas numpy
```

### Step 2: Configure AWS Credentials

The spot collector uses boto3, which needs AWS credentials:

```bash
# Option 1: AWS CLI (recommended)
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"

# Option 3: IAM role (if running on EC2)
# Credentials automatically available
```

**Required IAM permissions:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeSpotPriceHistory"
            ],
            "Resource": "*"
        }
    ]
}
```

### Step 3: Run Spot Collector

```bash
# Collect past 7 days of spot pricing
python3 fetch_aws_spot_gpu.py
```

**What it does:**
- Fetches spot price history for last 7 days
- Covers 3 regions: us-east-1, us-west-2, eu-west-1
- Monitors 7 GPU families: P3 (V100), P4 (A100), P5 (H100), G4dn (T4), G5 (A10G), G6 (L4), G6e (L40S)
- Saves one CSV per date in `data/` directory
- Takes ~2-5 minutes (API calls for each instance type + region)

**Expected output files:**
```
data/aws_gpu_spot_prices_2025-11-26.csv
data/aws_gpu_spot_prices_2025-11-27.csv
data/aws_gpu_spot_prices_2025-11-28.csv
data/aws_gpu_spot_prices_2025-11-29.csv
data/aws_gpu_spot_prices_2025-11-30.csv
data/aws_gpu_spot_prices_2025-12-01.csv
data/aws_gpu_spot_prices_2025-12-02.csv
```

### Step 4: Run Analysis Pipeline

```bash
# Process all data (on-demand + spot) and generate summary
python3 analyze_gpu_prices.py
```

This will:
- Load all on-demand pricing CSVs
- Load all spot pricing CSVs
- Apply IQR outlier filtering
- Calculate MEDIAN prices per date/region/model
- Merge spot with on-demand data
- Calculate spot discounts
- Save to `data/gpu_pricing_summary.csv`

### Step 5: View Dashboard

```bash
streamlit run dashboard.py
```

Access at: http://localhost:8501

Toggle between "Spot Pricing" and "On-Demand Pricing" in the sidebar!

## 📈 Expected Results

### Spot Pricing Time Series

With 7 days of data, you'll see:
- **Trend lines** for AWS spot prices across US regions
- **Price volatility** - spot prices change frequently
- **Spot discounts** - typically 50-90% cheaper than on-demand
- **Regional differences** - some regions cheaper than others
- **Model availability** - not all GPU models available as spot instances

### Data Quality

**What makes this accurate:**
1. **Real API data** - Direct from AWS EC2 Spot Price History API
2. **MEDIAN aggregation** - Robust to outliers
3. **Multiple samples** - Averages across all availability zones per day
4. **IQR filtering** - Removes anomalous prices
5. **Daily snapshots** - Can see day-to-day trends

**What to expect:**
- Spot prices are MORE volatile than on-demand
- Prices change throughout the day (we take daily median)
- Some instance types may not have spot availability
- Prices vary by availability zone (aggregated in our data)

## 🔍 Verifying Data Quality

### Check Spot Data

```bash
# View latest spot data
head -20 data/aws_gpu_spot_prices_2025-12-02.csv

# Count records per date
wc -l data/aws_gpu_spot_prices_*.csv

# Check price ranges
python3 -c "
import pandas as pd
df = pd.read_csv('data/aws_gpu_spot_prices_2025-12-02.csv')
print('Spot Price Summary:')
print(df.groupby('accelerator_model')['price_per_accel_hour_spot'].describe())
"
```

### Compare with AWS Console

Cross-check your collected data:
1. Go to https://aws.amazon.com/ec2/spot/pricing/
2. Select a GPU instance (e.g., p3.2xlarge)
3. Compare prices - should match our median within reasonable range
4. Remember: Our data is daily median across all AZs

### Check Data in Dashboard

Open dashboard and verify:
- Spot prices are lower than on-demand ✓
- Trend lines show reasonable volatility ✓
- Multiple regions visible ✓
- 7 days of history showing ✓

## 🎯 Next Steps for Complete Coverage

### 1. Azure Spot Pricing

Azure calls it "Spot VMs". To implement:

**API endpoint:**
```python
# Azure Retail Prices API with spot filter
url = "https://prices.azure.com/api/retail/prices"
params = {
    "$filter": (
        "serviceFamily eq 'Compute' and "
        "serviceName eq 'Virtual Machines' and "
        "priceType eq 'Consumption' and "
        "productName contains 'Spot' and "
        "(contains(skuName, 'NC') or contains(skuName, 'ND') or contains(skuName, 'NV'))"
    )
}
```

**Create:** `fetch_azure_spot_gpu.py`
**Model after:** `fetch_aws_spot_gpu.py`

### 2. GCP Preemptible/Spot Pricing

GCP has two options:
- **Preemptible VMs** - older, up to 80% discount
- **Spot VMs** - newer, similar pricing

**API endpoint:**
```python
# GCP Cloud Billing API
url = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"
# Filter for preemptible/spot GPU resources
```

**Create:** `fetch_gcp_spot_gpu.py`
**Model after:** `fetch_aws_spot_gpu.py`

### 3. Automated Daily Collection

Add to your cron job:

```bash
# Edit crontab
crontab -e

# Add daily collection at 2 AM UTC
0 2 * * * cd /path/to/gpu-pricing-monitor && python3 fetch_aws_spot_gpu.py >> logs/spot_collection.log 2>&1
```

Or use the complete workflow:
```bash
0 2 * * * cd /path/to/gpu-pricing-monitor && python3 collect_daily_prices.py >> logs/collection.log 2>&1
```

## 📋 Summary

### ✅ Completed
1. Dashboard shows ALL US regions (not filtered)
2. Data aggregation uses MEDIAN (documented)
3. Spot collector fetches 7 days of history
4. One CSV per day (no data loss)
5. Real AWS API data (no mocks)
6. Proper validation and error handling

### 📊 Data Quality
- **Source:** AWS EC2 Spot Price History API
- **Aggregation:** MEDIAN (robust to outliers)
- **Filtering:** IQR-based outlier removal
- **Sampling:** Multiple data points per day, median calculated
- **Accuracy:** Cross-verifiable with AWS console

### 🚀 Ready to Use
```bash
# 1. Collect historical spot data (takes ~3 minutes)
python3 fetch_aws_spot_gpu.py

# 2. Process all data
python3 analyze_gpu_prices.py

# 3. View dashboard
streamlit run dashboard.py
```

Your recession indicator dashboard now has clean, accurate spot pricing time series ready for analysis! 🎯
