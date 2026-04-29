# AuGR-R

This is the pytorch implementation for our paper AuGR on CTR datasets.

## Setup and data preparation

### 1) Download datasets

The CTR data zips are hosted in the reczoo/datasets repository:

- https://github.com/reczoo/datasets

Download the Avazu and/or TaobaoAd zip files from that repo and extract them locally.

### 2) Prepare the dataset folders

Each dataset folder should contain the CSV files and the model config JSON used by the
training scripts, for example:

```
Avazu_x4/
	train.csv
	valid.csv
	test.csv
	avazu_grouping_model_config_v1.json
TaobaoAd_x1/
	train.csv
	valid.csv
	test.csv
	taobao_grouping_model_config_v1.json
```

### 3) Generate vocab dictionaries

The training scripts expect a vocab JSON that you generate from the training CSV.
Use the indexing scripts:

```
python -m src.indexing.gen_avazu_vocab \
	--input_path /path/to/Avazu_x4/train.csv \
	--output_path /path/to/Avazu_x4/avazu_vocab.json

python -m src.indexing.gen_taobao_vocab \
	--input_path /path/to/TaobaoAd_x1/train.csv \
	--output_path /path/to/TaobaoAd_x1/taobao_vocab.json
```

### 4) Run training with the provided scripts

The training launchers live in [src/training/scripts](src/training/scripts). They are
intended to be run from the repo root and will `cd` internally as needed.

#### Avazu

```
bash src/training/scripts/run_train_avazu_ctr_v1.sh <exp_name>
```

Update these fields for your environment in
[src/training/scripts/run_train_avazu_ctr_v1.sh](src/training/scripts/run_train_avazu_ctr_v1.sh#L11-L84):

- `DATASET_ROOT` to your local Avazu_x4 folder.
- `TRAIN_DATA_PATH`, `VALID_DATA_PATH`, `TEST_DATA_PATH` if your CSV names differ.
- `AVAZU_VOCAB_PATH` (the vocab JSON generated in step 3).
- `MODEL_CONFIG_PATH` (the grouping config JSON located in the dataset folder).
- Optional: `WANDB_API_KEY` / `WANDB_PROJECT` / `WANDB_RUN_NAME` if you use W&B.

#### Taobao

```
bash src/training/scripts/run_train_taobao_ctr_v1.sh <exp_name>
```

Update these fields for your environment in
[src/training/scripts/run_train_taobao_ctr_v1.sh](src/training/scripts/run_train_taobao_ctr_v1.sh#L11-L95):

- `DATASET_ROOT` (or pass it as the second argument).
- `TRAIN_DATA_PATH`, `VALID_DATA_PATH`, `TEST_DATA_PATH` if your CSV names differ. Note that TaoBao dataset does not have its own validation data split, the base script provided uses test.csv for both VALID and TEST splits.
- `TAOBAO_VOCAB_PATH` (the vocab JSON generated in step 3).
- `MODEL_CONFIG_PATH` (the grouping config JSON located in the dataset folder).
- Optional: feature lists (`CATEGORICAL_FEATURES`, `ITEM_FEATURES`, `SEQUENCE_FEATURES`)
	if your CSV columns differ from the defaults.
- Optional: `WANDB_API_KEY` / `WANDB_PROJECT` / `WANDB_RUN_NAME` if you use W&B.