#!/usr/bin/env python3
"""
Test script to verify GPU pricing logic without downloading large file
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from fetch_aws_gpu import extract_gpu_prices
from utils import detect_aws_accel_model_and_count, save_to_csv

# Mock AWS pricing data structure for testing
mock_pricing_data = {
    "products": {
        "sku1": {
            "attributes": {
                "instanceType": "p3.2xlarge",
                "location": "US East (N. Virginia)",
                "tenancy": "Shared",
                "operatingSystem": "Linux"
            }
        },
        "sku2": {
            "attributes": {
                "instanceType": "g4dn.xlarge", 
                "location": "US East (N. Virginia)",
                "tenancy": "Shared",
                "operatingSystem": "Linux"
            }
        },
        "sku3": {
            "attributes": {
                "instanceType": "p5.48xlarge",
                "location": "US East (N. Virginia)", 
                "tenancy": "Shared",
                "operatingSystem": "Linux"
            }
        },
        "sku4": {
            "attributes": {
                "instanceType": "t3.micro",  # Non-GPU instance
                "location": "US East (N. Virginia)",
                "tenancy": "Shared", 
                "operatingSystem": "Linux"
            }
        }
    },
    "terms": {
        "OnDemand": {
            "sku1": {
                "term1": {
                    "priceDimensions": {
                        "dim1": {
                            "pricePerUnit": {"USD": "3.06"}
                        }
                    }
                }
            },
            "sku2": {
                "term2": {
                    "priceDimensions": {
                        "dim2": {
                            "pricePerUnit": {"USD": "0.526"}
                        }
                    }
                }
            },
            "sku3": {
                "term3": {
                    "priceDimensions": {
                        "dim3": {
                            "pricePerUnit": {"USD": "98.32"}
                        }
                    }
                }
            }
        }
    }
}

def test_gpu_detection():
    """Test GPU model detection logic"""
    print("Testing GPU detection...")
    test_cases = [
        ("p3.2xlarge", ("V100", 1)),
        ("p3dn.24xlarge", ("V100", 8)),
        ("p4d.24xlarge", ("A100", 8)),
        ("p5.48xlarge", ("H100", 8)),
        ("g4dn.xlarge", ("T4", 1)),
        ("g5.xlarge", ("A10G", 1)),
        ("t3.micro", None)
    ]
    
    for instance_type, expected in test_cases:
        result = detect_aws_accel_model_and_count(instance_type)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {instance_type} -> {result} (expected {expected})")

def test_price_extraction():
    """Test price extraction logic"""
    print("\nTesting price extraction...")
    region_name = "US East (N. Virginia)"
    df = extract_gpu_prices(mock_pricing_data, region_name)
    
    print(f"Found {len(df)} GPU instances:")
    if not df.empty:
        print(df.to_string(index=False))
        
        # Test specific values
        p3_row = df[df['instance_type'] == 'p3.2xlarge']
        if not p3_row.empty:
            print(f"\np3.2xlarge test:")
            print(f"  Price: ${p3_row.iloc[0]['price_hour_ondemand']}/hr")
            print(f"  GPU model: {p3_row.iloc[0]['gpu_model']}")
            print(f"  Price per GPU: ${p3_row.iloc[0]['price_per_gpu_hour']}/hr")
    else:
        print("No GPU instances found!")

def test_csv_output():
    """Test CSV saving"""
    print("\nTesting CSV output...")
    region_name = "US East (N. Virginia)"
    df = extract_gpu_prices(mock_pricing_data, region_name)
    
    if not df.empty:
        output_dir = Path("test_data")
        filepath = save_to_csv(df, output_dir, 'aws_gpu_prices_test')
        print(f"Saved test data to: {filepath}")
        
        # Verify file contents
        df_loaded = pd.read_csv(filepath)
        print(f"Verified: CSV contains {len(df_loaded)} rows")
        return filepath
    return None

if __name__ == "__main__":
    print("=== GPU Pricing Logic Test ===")
    test_gpu_detection()
    test_price_extraction() 
    test_csv_output()
    print("\n=== Test Complete ===")