#!/bin/bash

echo $#
DATA_DIR="$1"
OUTPUT_DIR="$2"
SAMPLE_SET_INDEX="$3"
TRAIN_CONFIG="$4"
SAMPLING_CONFIG="$5"
SEED="$6"

echo $DATA_DIR
echo $OUTPUT_DIR
echo $EXPERIMENT_NAME
echo $SAMPLE_SET_INDEX
echo $TRAIN_CONFIG
echo $SAMPLING_CONFIG
echo $SEED


echo "Starting task"

INIT_INDICES=$(sed "${SAMPLE_SET_INDEX}q;d" src/Configs/init_indices/prostate_indices)
echo "Init indices used: $INIT_INDICES"

python src/main.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR --data_config data_config/data_config_prostate.yaml --train_config al_config.yaml --train__train_indices $INIT_INDICES --model__out_channels 2 --train__loss__normalize_fct sigmoid --train__loss__n_classes 2 --train__val_plot_slice_interval 5 --train_config $TRAIN_CONFIG --sampling_config $SAMPLING_CONFIG --seed $SEED

echo "Reached end of job file."