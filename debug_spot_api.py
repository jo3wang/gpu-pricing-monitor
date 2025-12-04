#!/usr/bin/env python3
"""
Debug script to test AWS Spot Price API and identify why it returns empty results.
"""

import boto3
from datetime import datetime, timedelta

def test_spot_api():
    """Test various configurations of the spot price API."""

    region = "us-east-1"
    ec2 = boto3.client("ec2", region_name=region)

    print("=" * 80)
    print("AWS Spot Price API Debug Tests")
    print(f"Region: {region}")
    print("=" * 80)

    # Test 1: Most basic query - no filters at all
    print("\n[TEST 1] Most basic query - t3.micro with NO filters...")
    try:
        response = ec2.describe_spot_price_history(
            InstanceTypes=['t3.micro'],
            MaxResults=5
        )
        spot_prices = response.get('SpotPrices', [])
        print(f"  Found {len(spot_prices)} spot prices")
        if spot_prices:
            for i, price in enumerate(spot_prices[:3]):
                print(f"  [{i+1}] ${price['SpotPrice']} - {price['AvailabilityZone']} - {price['Timestamp']} - {price['ProductDescription']}")
        else:
            print("  ✗ No data returned")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Check account spot instance permissions
    print("\n[TEST 2] Checking account permissions for spot instances...")
    try:
        # Try to describe spot instance requests (should work even if no requests exist)
        response = ec2.describe_spot_instance_requests(MaxResults=5)
        print(f"  ✓ Account has spot instance API access")
        print(f"  Spot requests found: {len(response.get('SpotInstanceRequests', []))}")
    except Exception as e:
        print(f"  ✗ Error accessing spot instance API: {e}")

    # Test 3: Try with very recent time range (last hour)
    print("\n[TEST 3] Testing with very recent time range (last 1 hour)...")
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        response = ec2.describe_spot_price_history(
            InstanceTypes=['t3.micro'],
            StartTime=start_time,
            EndTime=end_time,
            MaxResults=10
        )
        spot_prices = response.get('SpotPrices', [])
        print(f"  Time range: {start_time} to {end_time}")
        print(f"  Found {len(spot_prices)} spot prices")
        if spot_prices:
            print(f"  Sample: ${spot_prices[0]['SpotPrice']} in {spot_prices[0]['AvailabilityZone']}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 4: Get spot price for ANY instance without specifying type
    print("\n[TEST 4] Getting spot prices WITHOUT specifying instance type...")
    try:
        response = ec2.describe_spot_price_history(
            MaxResults=10
        )
        spot_prices = response.get('SpotPrices', [])
        print(f"  Found {len(spot_prices)} spot prices")
        if spot_prices:
            print(f"  Recent spot prices:")
            for i, price in enumerate(spot_prices[:5]):
                print(f"    [{i+1}] {price['InstanceType']}: ${price['SpotPrice']} - {price['AvailabilityZone']} - {price['Timestamp']}")
        else:
            print("  ✗ No data returned - This suggests an account limitation")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 5: Try different region
    print("\n[TEST 5] Testing in us-west-2 region...")
    try:
        ec2_west = boto3.client("ec2", region_name="us-west-2")
        response = ec2_west.describe_spot_price_history(
            InstanceTypes=['t3.micro'],
            MaxResults=5
        )
        spot_prices = response.get('SpotPrices', [])
        print(f"  Found {len(spot_prices)} spot prices in us-west-2")
        if spot_prices:
            print(f"  Sample: ${spot_prices[0]['SpotPrice']} in {spot_prices[0]['AvailabilityZone']}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 6: Check AWS credentials and region config
    print("\n[TEST 6] Checking AWS configuration...")
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"  Account ID: {identity['Account']}")
        print(f"  User ARN: {identity['Arn']}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("-" * 80)
    print("If ALL tests return 0 results, this indicates one of the following:")
    print("1. AWS Free Tier accounts may have limited access to spot price history")
    print("2. New AWS accounts may need to request spot instance permissions")
    print("3. There may be a region-specific or timing issue with the API")
    print("\nRECOMMENDATION:")
    print("Try creating a spot instance request through the AWS Console to")
    print("verify your account has spot instance access enabled.")
    print("=" * 80)

if __name__ == "__main__":
    test_spot_api()
