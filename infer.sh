export VERSION=bidirectional
export SAVED_MODEL_PATH=models/14rest/best_test_model.pt
export TEXT="Great food at REASONABLE prices , makes for an evening that ca n't be beat !"
#TODO: Training
python main.py \
  --input TEXT \
  --version $VERSION \
  --model_file_path $SAVED_MODEL_PATH
