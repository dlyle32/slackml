#!/bin/bash

sed -i '' "s|.*UserData.*|          \"UserData\": \"base64_encoded_bash_script\",|g" spot_fleet_config_p3.json
USER_DATA=`base64 scripts/user_data_script.sh -b0`
sed -i '' "s|base64_encoded_bash_script|$USER_DATA|g" spot_fleet_config_p3.json
#aws ec2 request-spot-fleet --spot-fleet-request-config file://spot_fleet_config_p3.json