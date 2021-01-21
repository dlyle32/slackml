#!/bin/bash
# Get instance ID, Instance AZ, Volume ID and Volume AZ 
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
AWS_REGION=us-east-2

VOLUME_ID=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=slack-data-checkpoints" --query "Volumes[].VolumeId" --output text)
VOLUME_AZ=$(aws ec2 describe-volumes --region $AWS_REGION --filter "Name=tag:Name,Values=slack-data-checkpoints" --query "Volumes[].AvailabilityZone" --output text)

# Proceed if Volume Id is not null or unset
if [ $VOLUME_ID ]; then
		# Check if the Volume AZ and the instance AZ are same or different.
		# If they are different, create a snapshot and then create a new volume in the instance's AZ.
		if [ $VOLUME_AZ != $INSTANCE_AZ ]; then
				SNAPSHOT_ID=$(aws ec2 create-snapshot \
						--region $AWS_REGION \
						--volume-id $VOLUME_ID \
						--description "`date +"%D %T"`" \
						--tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=slack-data-checkpoints-snapshot}]' \
						--query SnapshotId --output text)

				aws ec2 wait --region $AWS_REGION snapshot-completed --snapshot-ids $SNAPSHOT_ID
				aws ec2 --region $AWS_REGION  delete-volume --volume-id $VOLUME_ID

				VOLUME_ID=$(aws ec2 create-volume \
						--region $AWS_REGION \
								--availability-zone $INSTANCE_AZ \
								--snapshot-id $SNAPSHOT_ID \
						--volume-type gp2 \
						--tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=slack-data-checkpoints}]' \
						--query VolumeId --output text)
				aws ec2 wait volume-available --region $AWS_REGION --volume-id $VOLUME_ID
		fi
		# Attach volume to instance
		aws ec2 attach-volume \
			--region $AWS_REGION --volume-id $VOLUME_ID \
			--instance-id $INSTANCE_ID --device /dev/sdf
		sleep 10

		# Mount volume and change ownership, since this script is run as root
		mkdir /training
		mount /dev/xvdf /training
		chown -R ubuntu: /training/
		cd /home/ubuntu/

		# Get training code
		git clone https://github.com/dlyle32/slackml.git
		chown -R ubuntu: slackml
		cd slackml

		# Initiate training using the tensorflow_36 conda environment
		#sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate tensorflow2_latest_p37;python src/train_lm.py --numepochs 40 --seqlength 40 --hiddensize 175 --dropoutrate 0.1 --regfactor 0 --datacap 30000 --learningrate 0.001 --minibatchsize 128 --modelbuilder kwlm_per_message.PerMessageLanguageModelBuilder --step 8 --optimizer adam --embedding --embeddingsize 512 --decayrate 0.95 --decaysteps 1000 --loadmodel /training/checkpoints/20201125/nodle_char_model.1606329829.029.h5 "
#		sudo /home/ubuntu/anaconda3/envs/tensorflow2_latest_p37/bin/pip uninstall -y tensorflow-gpu tensorflow-cpu tensorflow-estimator tensorflow-serving-api
#		sudo /home/ubuntu/anaconda3/envs/tensorflow2_latest_p37/bin/pip install tf-nightly
		sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate tensorflow2_latest_p37;  python src/train_lm.py --datadir data/user_msgs/ --numepochs 12 --seqlength 40  --hiddensize 128 --dropoutrate 0.1 --regfactor 0 --datacap 60000 --learningrate 0.001 --minibatchsize 64 --modelbuilder kattn_lm.AttentionModelBuilder --step 8 --optimizer adam --decayrate 0.95 --decaysteps 800  --ffdim 1024 --attention_heads 4 "
fi

# After training, clean up by cancelling spot requests and terminating itself
SPOT_FLEET_REQUEST_ID=$(aws ec2 describe-spot-instance-requests --region $AWS_REGION --filter "Name=instance-id,Values='$INSTANCE_ID'" --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" --output text)
aws ec2 cancel-spot-fleet-requests --region $AWS_REGION --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID --terminate-instances
