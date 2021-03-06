# B-MRC
MRC approach for Aspect-based Sentiment Analysis (ABSA)

<img src="docs/imgs/bmrc.png" width="45%" height="45%" align="center" />

**Paper:** [Bidirectional Machine Reading Comprehension for Aspect Sentiment Triplet Extraction](https://arxiv.org/abs/2103.07665)

**Dataset:** [https://github.com/xuuuluuu/SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data)

### Usage

- Prepare data:
```commandline
python data_process.py --data_path data/14lap --version bidirectional (unidirectional)

Arguments:
    --data_path :       Path to the dataset
    --version   :       Optional version: unidirectional (A2O) and bidirectional (A2O + O2A) 
                        (default = 'bidirectiona')
                        Choices=['uni', 'bi', 'unidirectional', 'bidirectional']
```

```commandline
python make_data_dual --data_path data/14lap/preprocess --version bidirectional (unidirectional)

Arguments:
    --data_path :       Path to the dataset
    --version   :       Optional version: unidirectional (A2O) and bidirectional (A2O + O2A)
                        (default = 'bidirectiona')
                        Choices=['uni', 'bi', 'unidirectional', 'bidirectional']
```

```commandline
python make_data_standard --data_path data/14lab/pair --output_path ./data/14lap/preprocess

Arguments:
    --data_path  :      Path to the dataset
    --output_path:      Path to the output data      
```

- Training:
```commandline
python main.py \
    --version bidirectional (unidirectional) \
    --data_path ./data/14lap/preprocess/ \
    --mode train \
    --model_type bert-base-uncased \
    --epoch_num 40 \
    --batch_size 4 \
    --learning_rate 1e-3
```

- Inference:
```commandline
python inference \
    --input TEXT \
    --model_file_path PATH_TO_MODEL \
    --version bidirectional/ unidirectional
```


- Using from bash:

```commandline
Training:
>>> bash train.sh

Test:
>>> bash test.sh

Inference:
>>> bash infer.sh
```