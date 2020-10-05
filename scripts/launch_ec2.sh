#!/bin/bash

aws ec2 run-instances \
    --image-id ami-0f379761b32319822 \
    --security-group-ids sg-0bf2a99d8db8b698b \
    --count 1 \
    --instance-type m4.xlarge \
    --key-name dlyle-slackai-c5lg \
    --subnet-id subnet-502ff93b \
    --query "Instances[0].InstanceId"
