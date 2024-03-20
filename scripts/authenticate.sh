#!/bin/bash

# Check if both access key ID and secret access key are provided
if [ $# -ne 2 ] && [ $# -ne 3 ]; then
echo "Usage: $0 {aws_access_key_id} {aws_secret_access_key} [wandb_access_key]"
    exit 1
fi

# Check if conf/local directory exists
if [ ! -d "conf/local" ]; then
    echo "Error: conf/local directory does not exist. Make sure your in the root of the project."
    exit 1
fi

# Prepare the YAML content
cat << EOF > conf/local/credentials.yaml
aws_access:
  client_kwargs:
    aws_access_key_id: $1
    aws_secret_access_key: $2
    region_name: "eu-west-2"
wandb_access: $3
EOF

echo "Credentials written to conf/local/credentials.yaml"
