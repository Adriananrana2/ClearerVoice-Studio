#!/bin/sh

#####
# Modify these lines
gpu_id=0,1													# Visible GPUs
n_gpu=2														# Number of GPU used for training
checkpoint_dir=''											# Leave empty if it's a new training, otherwise provide the name as 'checkpoints/log_...'
config_pth=config/config_MADEEG_eeg_neuroheed.yaml			# The config file, only used if it's a new training
#####


# create checkpoint log folder
if [ -z ${checkpoint_dir} ]; then
	checkpoint_dir='checkpoints/log_'$(date '+%Y-%m-%d(%H:%M:%S)')
	train_from_last_checkpoint=0
	mkdir -p ${checkpoint_dir}
	cp $config_pth ${checkpoint_dir}/config.yaml
else
	train_from_last_checkpoint=1
	config_pth=${checkpoint_dir}/config.yaml
fi
yaml_name=log_$(date '+%Y-%m-%d(%H:%M:%S)')
cat $config_pth > ${checkpoint_dir}/${yaml_name}.txt

# call training
export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port=$(date '+88%S') \
train.py \
--config $config_pth \
--checkpoint_dir $checkpoint_dir \
--train_from_last_checkpoint $train_from_last_checkpoint \
>>${checkpoint_dir}/$yaml_name.txt 2>&1
