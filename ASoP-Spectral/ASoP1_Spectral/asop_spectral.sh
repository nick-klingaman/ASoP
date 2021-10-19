#!/bin/bash

source $CONDA_SOURCE
conda activate $CONDA_ENV_ROOT/_CMEC_asop_spectral

printf "\nlog: $CMEC_WK_DIR/asop_spectral.log.txt\n"

cd $CMEC_WK_DIR
if [ "$CMEC_OBS_DATA" == "None" ];
then
    python $CMEC_CODE_DIR/ASoP1_spectral_cmec_workflow.py \
    $CMEC_MODEL_DATA $CMEC_WK_DIR \
    --config $CMEC_CONFIG_DIR/cmec.json >> $CMEC_WK_DIR/asop_spectral.log.txt
else
    python $CMEC_CODE_DIR/ASoP1_spectral_cmec_workflow.py \
    $CMEC_MODEL_DATA $CMEC_WK_DIR --obs_dir  $CMEC_OBS_DATA \
    --config $CMEC_CONFIG_DIR/cmec.json >> $CMEC_WK_DIR/asop_spectral.log.txt
fi
