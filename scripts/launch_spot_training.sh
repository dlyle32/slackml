#!/bin/bash

sed -i "" "s|AWS_REGION=.*|AWS_REGION=$2|g" scripts/user_data_script.sh
sed -i '' "s|.*UserData.*|          \"UserData\": \"base64_encoded_bash_script\",|g" $1
USER_DATA=`base64 scripts/user_data_script.sh -b0`
sed -i '' "s|base64_encoded_bash_script|$USER_DATA|g" $1
aws ec2 request-spot-fleet --region $2 --spot-fleet-request-config file://$1
