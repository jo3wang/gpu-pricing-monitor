# GPU Pricing Monitor

A Python tool to fetch and monitor AWS EC2 GPU instance pricing.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script:**
   ```bash
   python fetch_aws_gpu.py
   ```

## Output

The script will:
- Download AWS EC2 pricing data for `us-east-1`
- Filter to GPU instance types (p3, p4, p5, g4, g5 families)
- Calculate price per GPU hour
- Save results to `data/aws_gpu_prices_YYYY-MM-DD.csv`

## CSV Columns

- `timestamp_utc`: ISO timestamp of when data was fetched
- `region`: AWS region (e.g., "us-east-1")
- `instance_type`: EC2 instance type (e.g., "p3.2xlarge")
- `gpu_model`: GPU model (V100, A100, H100, T4, A10G)
- `num_gpus`: Number of GPUs per instance
- `price_hour_ondemand`: On-demand price per hour (USD)
- `price_per_gpu_hour`: Price per GPU per hour (USD)

## GPU Instance Mapping

- `p3.*` → V100 (1 GPU)
- `p3dn.*` → V100 (8 GPUs)
- `p4d.*` / `p4de.*` → A100 (8 GPUs)
- `p5.*` → H100 (8 GPUs)
- `g4dn.*` → T4 (1 GPU)
- `g5.*` → A10G (1 GPU)

