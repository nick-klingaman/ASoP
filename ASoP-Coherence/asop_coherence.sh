#!/bin/bash

source $CONDA_SOURCE
conda activate $CONDA_ENV_ROOT/_CMEC_asop_coherence

printf "\nlog: $CMEC_WK_DIR/asop_coherence.log.txt\n"

cd $CMEC_WK_DIR

python $CMEC_CODE_DIR/ASoP_coherence_cmec_workflow.py \
$CMEC_MODEL_DATA $CMEC_OBS_DATA $CMEC_WK_DIR \
--config $CMEC_CONFIG_DIR/cmec.json >> $CMEC_WK_DIR/asop_coherence.log.txt

