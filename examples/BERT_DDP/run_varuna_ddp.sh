#!/bin/bash

# Script to pretrain BERT on the wikicorpus dataset with varuna

CMD="python3 -m varuna.run_varuna --no_morphing"

# for Varuna
machine_list="list"
gpus_per_node=4

batch_size_phase1=1024 #This is in total, for all GPUs (for pretraining should use 65536)
chunk_size=16 # getting OOM error with batch size 64 in V100-16GB
nstages=1
CODEDIR="/home/fot/DeepLearningExamples/PyTorch/LanguageModeling/BERT" # fix this

CMD+=" --machine_list=$machine_list"
CMD+=" --gpus_per_node=$gpus_per_node"
CMD+=" --batch_size=$batch_size_phase1"
CMD+=" --chunk_size=$chunk_size"
CMD+=" --nstages=$nstages"
CMD+=" --code_dir=$CODEDIR"
CMD+=" run_pretraining.py"

CMD_BASE=$CMD

# for BERT 
BERT_PREP_WORKING_DIR="/home/fot/BERT/"
BERT_CONFIG=$CODEDIR/bert_config.json
RESULTS_DIR=$CODEDIR/results_bert
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints
seed=12439

# phase 1
DATASET1=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
DATA_DIR_PHASE1=$BERT_PREP_WORKING_DIR/${DATASET1}
train_steps_phase1=7038
warmup_proportion_phase1="0.2843"
learning_rate_phase1="6e-3"

# phase 2
DATASET2=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
DATA_DIR_PHASE2=$BERT_PREP_WORKING_DIR/${DATASET2}
train_steps_phase2=1563
warmup_proportion_phase2="0.128"
learning_rate_phase1="4e-3"

# ------------ run phase 1
mkdir -p $CHECKPOINTS_DIR

CMD=$CMD_BASE
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --max_steps=$train_steps_phase1"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --warmup_proportion=$warmup_proportion_phase1"
CMD+=" --learning_rate=$learning_rate_phase1"
CMD+=" --seed=$seed"
CMD+=" --do_train"
CMD+=" --varuna"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "

echo "Run phase 1"
echo $CMD
$CMD


# ------------ run phase 2

#CMD=$CMD_BASE
#CMD+=" --input_dir=$DATA_DIR_PHASE2"
#CMD+=" --max_steps=$train_steps_phase2"
#CMD+=" --config_file=$BERT_CONFIG"
#CMD+=" --bert_model=bert-large-uncased"
#CMD+=" --max_seq_length=512"
#CMD+=" --max_predictions_per_seq=80"
#CMD+=" --warmup_proportion=$warmup_proportion_phase2"
#CMD+=" --learning_rate=$learning_rate_phase2"
#CMD+=" --seed=$seed"
#CMD+=" --do_train --phase2" # TODO: add this here: --resume_from_checkpoint --phase1_end_step=$train_steps"
#CMD+=" --varuna"
#CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "

#echo "Run phase 2"
#$CMD

