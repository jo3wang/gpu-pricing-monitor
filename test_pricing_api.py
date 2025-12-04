#!/usr/bin/env python3
"""
Test AWS Pricing API as alternative to spot price history API.
The Pricing API provides current on-demand pricing and may have spot pricing info.
"""

import boto3
import json

def test_pricing_api():
    """Test AWS Pricing API for GPU instance pricing."""

    print("=" * 80)
    print("Testing AWS Pricing API")
    print("=" * 80)

    # Pricing API is only available in us-east-1 and ap-south-1
    pricing = boto3.client('pricing', region_name='us-east-1')

    # Test 1: Get pricing for p3.2xlarge
    print("\n[TEST 1] Getting pricing for p3.2xlarge...")
    try:
        response = pricing.get_products(
            ServiceCode='AmazonEC2',
            Filters=[
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'instanceType',
                    'Value': 'p3.2xlarge'
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'location',
                    'Value': 'US East (N. Virginia)'
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'operatingSystem',
                    'Value': 'Linux'
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'tenancy',
                    'Value': 'Shared'
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'preInstalledSw',
                    'Value': 'NA'
                }
            ],
            MaxResults=1
        )

        if response['PriceList']:
            price_data = json.loads(response['PriceList'][0])
            print(f"  ✓ Found pricing data for p3.2xlarge")

            # Extract on-demand pricing
            if 'terms' in price_data and 'OnDemand' in price_data['terms']:
                for term_key, term_data in price_data['terms']['OnDemand'].items():
                    for dimension_key, dimension_data in term_data['priceDimensions'].items():
                        price = dimension_data['pricePerUnit']['USD']
                        print(f"  On-demand price: ${price}/hour")

            # Check if spot pricing is available
            if 'terms' in price_data:
                print(f"  Available terms: {list(price_data['terms'].keys())}")
        else:
            print("  ✗ No pricing data found")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 2: Try to find spot pricing information
    print("\n[TEST 2] Checking if Pricing API includes spot pricing...")
    try:
        response = pricing.get_products(
            ServiceCode='AmazonEC2',
            Filters=[
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'instanceType',
                    'Value': 'g4dn.xlarge'
                },
                {
                    'Type': 'TERM_MATCH',
                    'Field': 'location',
                    'Value': 'US East (N. Virginia)'
                }
            ],
            MaxResults=5
        )

        print(f"  Found {len(response['PriceList'])} price entries")

        for i, price_str in enumerate(response['PriceList'][:2]):
            price_data = json.loads(price_str)
            product_family = price_data['product'].get('productFamily', 'N/A')
            usage_type = price_data['product']['attributes'].get('usagetype', 'N/A')
            print(f"  [{i+1}] Product family: {product_family}, Usage type: {usage_type}")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("-" * 80)
    print("The Pricing API only provides ON-DEMAND pricing, not spot pricing.")
    print("Spot pricing must come from describe_spot_price_history API.")
    print("\nFor new AWS accounts, you may need to:")
    print("1. Create a spot instance request through AWS Console first")
    print("2. Wait 24-48 hours for account verification")
    print("3. Contact AWS Support to enable spot pricing access")
    print("=" * 80)

if __name__ == "__main__":
    test_pricing_api()
