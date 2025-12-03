#!/usr/bin/env python3
"""
Daily GPU Pricing Collection Workflow
Orchestrates data collection from all cloud providers (AWS on-demand, AWS spot, Azure, GCP)
and runs the analysis pipeline to generate the summary CSV.
"""

import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def run_collector(script_name: str) -> bool:
    """
    Run a pricing collector script.

    Args:
        script_name: Name of the Python script to run

    Returns:
        True if successful, False if failed
    """
    logging.info(f"=" * 80)
    logging.info(f"Running {script_name}...")
    logging.info(f"=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per collector
        )

        # Log stdout
        if result.stdout:
            logging.info(f"{script_name} output:\n{result.stdout}")

        # Log stderr if present
        if result.stderr:
            logging.warning(f"{script_name} stderr:\n{result.stderr}")

        if result.returncode != 0:
            logging.error(f"{script_name} failed with exit code {result.returncode}")
            return False

        logging.info(f"{script_name} completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logging.error(f"{script_name} timed out after 10 minutes")
        return False
    except Exception as e:
        logging.error(f"Error running {script_name}: {e}")
        return False


def main():
    """Main workflow to collect all GPU pricing data."""
    start_time = datetime.utcnow()
    logging.info("=" * 80)
    logging.info("STARTING DAILY GPU PRICING COLLECTION")
    logging.info(f"Timestamp: {start_time.isoformat()}Z")
    logging.info("=" * 80)

    # Define collectors to run in order
    collectors = [
        "fetch_aws_gpu.py",          # AWS on-demand GPU pricing
        "fetch_aws_spot_gpu.py",     # AWS spot GPU pricing (NEW!)
        "fetch_azure_gpu.py",        # Azure pay-as-you-go GPU pricing
        "fetch_gcp_accel.py",        # GCP GPU and TPU pricing
    ]

    # Track results
    results = {}

    # Run each collector
    for collector in collectors:
        success = run_collector(collector)
        results[collector] = success

    logging.info("=" * 80)
    logging.info("COLLECTION SUMMARY")
    logging.info("=" * 80)

    # Print results
    for collector, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logging.info(f"{collector:30s} {status}")

    # Check if all succeeded
    all_success = all(results.values())
    failed_count = sum(1 for s in results.values() if not s)

    if all_success:
        logging.info("All collectors completed successfully!")
    else:
        logging.warning(f"{failed_count} collector(s) failed")

    logging.info("=" * 80)
    logging.info("Running analysis pipeline to generate summary...")
    logging.info("=" * 80)

    # Run the analysis pipeline to process all collected data
    try:
        analysis_success = run_collector("analyze_gpu_prices.py")

        if analysis_success:
            logging.info("Analysis pipeline completed successfully!")
            logging.info("Summary CSV generated: data/gpu_pricing_summary.csv")
        else:
            logging.error("Analysis pipeline failed")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error running analysis pipeline: {e}")
        sys.exit(1)

    # Final summary
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    logging.info("=" * 80)
    logging.info("DAILY COLLECTION COMPLETE")
    logging.info(f"Duration: {duration:.1f} seconds")
    logging.info(f"End time: {end_time.isoformat()}Z")
    logging.info("=" * 80)

    # Exit with appropriate code
    if all_success and analysis_success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
