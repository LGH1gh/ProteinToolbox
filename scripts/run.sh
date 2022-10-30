TOKENIZER_NAME=OneHot
ENCODER_NAME=Resnet
DATASET_NAME=TAPE_secondary_structure
DATA_DIR=./benchmark/


python -u main.py \
  --accelerator gpu \
  --devices 1, \
  --max_epochs 10 \
  --tokenizer-name $TOKENIZER_NAME \
  --encoder-name $ENCODER_NAME \
  --dataset-name $DATASET_NAME \
  --data-dir $DATA_DIR \
  --hidden-layer-num 3 \
  --hidden-dim 512 \
  --activation gelu \
  --dropout 0.1 \
  --worker-num 8 \
  --task-test-name cb513 \
  --task-type ss3 \
  --in-memory False \
  --train-batch-size 8 \
  --inference-batch-size 8

