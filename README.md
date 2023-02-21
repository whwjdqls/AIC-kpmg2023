# AIC-kpmg2023

Leveraging pretrained models from [KoELECTRA](https://github.com/monologg/KoELECTRA/tree/master/pretrain) and adapting to train on the KorQuAD 2.1 dataset. Specifically,
- We added data preprocessing
- We modified the transformer to fit the KorQuAD 2.1 dataset
- We implemented the sliding window in long context to improve accuracy
- We created our own Q&A datasets on business report and used them for training

## Preparation
- The koelectra finetuning is performed by referring to [this link](https://github.com/monologg/KoELECTRA)
- The transformer can be directly used through [this huggingface link](https://github.com/huggingface/transformers)
- You can download the KorQuAD 2.1 dataset in [this link](https://korquad.github.io/)

## Training/Validation
You can just clone the [KoELECTRA](https://github.com/monologg/KoELECTRA) repo into your own computer. Then, overwrite our files in the `KoELECTR/finetune` directory.
<br><br>
To train this model run:
```
python run_squad.py --task korquad --config_file koelectra-base-v3.json
```
To validate this model run:
```
python run_squad.py --task korquad --config_file koelectra-base-v3_test.json
```
## Making Custom QA Dataset
Making custom dataset in the form of KorQuAD 2.1 form target files
```
python make_custom_dataset.py --data_dir {directory containing html files} --name 정빈
```
use name for distinguishing people when more than one are making dataset. (for unique id)
