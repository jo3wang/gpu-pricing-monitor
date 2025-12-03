#!/bin/bash

# GPU Pricing Monitor - Daily Data Collection Script
# This script collects fresh data from AWS and Azure, processes it, and updates the dashboard

echo "Starting daily GPU pricing data collection at $(date)"

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Function to run a command and check for errors
run_with_error_check() {
    echo "Running: $1"
    if eval "$1"; then
        echo "✓ Success: $1"
        return 0
    else
        echo "✗ Failed: $1"
        return 1
    fi
}

# Collect AWS data
echo "Collecting AWS GPU pricing data..."
if ! run_with_error_check "python fetch_aws_gpu.py"; then
    echo "AWS collection failed, continuing with Azure..."
fi

# Collect Azure data
echo "Collecting Azure GPU pricing data..."
if ! run_with_error_check "python fetch_azure_gpu.py"; then
    echo "Azure collection failed, continuing with analysis..."
fi

# Optional: Collect GCP data if script exists
if [ -f "fetch_gcp_accel.py" ]; then
    echo "Collecting GCP accelerator pricing data..."
    run_with_error_check "python fetch_gcp_accel.py"
fi

# Process and analyze the collected data
echo "Processing collected data and computing stress index..."
if ! run_with_error_check "python analyze_gpu_prices.py"; then
    echo "Data analysis failed!"
    exit 1
fi

echo "Daily data collection completed successfully at $(date)"
echo "Dashboard data updated. Access at: http://localhost:8501"

# Optional: Send notification (uncomment if you want email notifications)
# echo "GPU pricing data updated on $(date)" | mail -s "GPU Pricing Monitor Update" your-email@example.com