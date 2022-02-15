export VERSION=bidirectional
export PREPROCESS_DATA_PATH=data/14rest/preprocess
export SAVED_MODEL_PATH=models/best_model.pt

#TODO: Training
python main.py \
  --data_path $PREPROCESS_DATA_PATH \
  --version $VERSION \
  --mode test \
  --save_model_path $SAVED_MODEL_PATH
