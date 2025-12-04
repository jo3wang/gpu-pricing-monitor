# AWS Spot Pricing Access Setup

## Problem
New AWS Free Tier accounts may not have immediate access to spot price history via the API, even though the AWS website shows spot pricing exists.

## Solution: Activate Spot Access

### Step 1: Request Spot Instance Limits
1. Go to AWS EC2 Console: https://console.aws.amazon.com/ec2/
2. Navigate to: **Limits** → **Request limit increase**
3. Request spot instance limits for your account (even if you don't plan to use them immediately)

### Step 2: Create a Test Spot Request (Activate Spot Access)
1. Go to: https://console.aws.amazon.com/ec2/home?region=us-east-1#SpotInstances:
2. Click **"Request Spot Instances"**
3. Configure a minimal request:
   - AMI: Any Amazon Linux 2 AMI
   - Instance type: **t3.micro** (Free Tier eligible)
   - Max price: $0.01/hour (well above typical spot price to ensure it launches)
   - Request type: **One-time** (not persistent)
   - Valid until: 1 hour from now
4. **Review and launch** the request
5. Once you see it appear in "Spot Requests", you can **cancel it immediately**
   - Select the request → **Actions** → **Cancel spot request**
   - Check "Terminate instances" if any launched

### Step 3: Wait and Retry API
After creating your first spot request:
1. Wait **15-30 minutes** for AWS to process the activation
2. Retry the spot pricing collection script:
   ```bash
   python debug_spot_api.py
   ```
3. You should now see spot pricing data returned

## Alternative: Check Service Quotas

1. Go to: https://console.aws.amazon.com/servicequotas/
2. Navigate to: **AWS services** → **Amazon Elastic Compute Cloud (EC2)**
3. Search for: "spot"
4. Check your spot instance quotas:
   - "All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests"
   - If quotas are 0, request an increase

## Alternative: Contact AWS Support

If the above doesn't work:
1. Go to: https://console.aws.amazon.com/support/
2. Create a case: **Service limit increase**
3. Request: "Enable spot price history API access for new account"

## Verify Access

Once activated, test with:
```bash
python debug_spot_api.py
```

Expected output: Should show spot prices for t3.micro and GPU instances.

## Timeline
- Spot request creation: **Immediate**
- API access activation: **15-30 minutes** after first spot request
- Support ticket response: **1-2 business days**
