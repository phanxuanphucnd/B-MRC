# B-MRC
MRC approach for Aspect-based Sentiment Analysis (ABSA)

<img src="docs/imgs/bmrc.png" width="35%" height="35%" align="center" />


### Usage

```commandline
python data_process.py --data_path data/14lap --a2o --o2a

Arguments:
    --data_path :       Path to the dataset
    --a2o       :       If use aspect to opions (forward) unidirectional
    --o2a       :       If use opinion to aspects (backward) unidirectional
    
    If contains --a2o and --o2a, use bidirectional.
```

```commandline
python make_data_dual --data_path data/14lap/processed --a2o --o2a

Arguments:
    --data_path :       Path to the dataset
    --a2o       :       If use aspect to opions (forward) unidirectional
    --o2a       :       If use opinion to aspects (backward) unidirectional
    
    If contains --a2o and --o2a, use bidirectional.
```

