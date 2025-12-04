#!/bin/bash
# Create IAM role for Spot Fleet

echo "Creating IAM role for Spot Fleet..."

# Create trust policy
cat > /tmp/spot-fleet-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "spotfleet.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the IAM role
aws iam create-role \
  --role-name aws-ec2-spot-fleet-tagging-role \
  --assume-role-policy-document file:///tmp/spot-fleet-trust-policy.json

# Attach the AWS managed policy for Spot Fleet
aws iam attach-role-policy \
  --role-name aws-ec2-spot-fleet-tagging-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole

echo "Role created successfully!"
echo "Role ARN: arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/aws-ec2-spot-fleet-tagging-role"
