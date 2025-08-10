## Overview:
This repository is replicating the BioSyn repository with improvements

## Setup

For setup, you can use Conda or pip with venv, for windows I recommend using Conda and for linux with gpu I recommend using pip with venv 

## Setup using pip and venv

### Create virtual environment

```bash
$ python -m venv myenv
```

### Activate virtual environment (windows)

```bash
$ ./myenv/Scripts/activate
```

### Activate virtual environment (linux)

```bash
$ source ./myenv/bin/activate
```

### Install requirements


```bash
$ pip install torch
$ pip install tqdm transformers requests
$ pip install faiss-cpu
```

If you're using linux and cuda, you can install faiss-gpu instead of faiss-cpu
```bash
$ conda install faiss-gpu -c pytorch
```


## Setup using Conda

### Create virtual environment:

```bash
$ conda create -n bsn python=3.9 -y
$ conda activate bsn
```

## Install torch:

Gpu use:

```bash
$ conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

CPU use:

```bash
$ conda install pytorch=2.6.0 cpuonly -c pytorch
```

## Remaining packages
```bash
$ conda install numpy tqdm transformers requests  -c conda-forge
$ conda install faiss-cpu -c conda-forge
```

If you are using linux and cuda, you can install faiss-gpu instead of faiss-cpu:
```bash
$ conda install faiss-gpu -c pytorch
```



## Download data

You can choose to download data of the datasets:
ncbi-disease, bc5dr-disease, bc5dr-chemical

if you want to download ncbi-disease:
```bash
$ python  download_ds.py --ds_name ncbi-disease
```

Change the arg --ds_name for other dataset
The data will be downloaded into folder data/ncbi-disease-normal folder
it will contain:
- processed_dev/
- processed_test/
- processed_train/
- processed_traindev/
- dev_dictionary.txt
- test_dictionary.txt
- train_dictionary.txt

<!-- ## Use fair data evaluation

In the folder data/data-ncbi-fair we have training/testing data for fair training and evaluation 
In the folder data/ncbi-disease-normal (could be the data you download using download_ds.py) having the normal data biosyn is training on -->

## Train

To train the model you have to execute train.py specifying the arguments:
(All arguments have default values where you can see the default in the train.py global variables in order to be able to run `python train.py` only, but also you can override them by specifying them, for example `python train.py --use_cuda`)


| Argument Name              | Description                         |
|----------------------------|-------------------------------------|
| `--model_name_or_path`     | Directory for pretrained model (we use default:  'dmis-lab/biobert-base-cased-v1.1')     |
| `--train_dictionary_path`  | Train dictionary path (path of .txt containing the dictionary)              |
| `--train_dir`              | Training set directory (dir containing the .concept files)              |
| `--output_dir`             | Directory for output trained model                |
| `--max_length`             | Max sequence length for tokenizer   |
| `--seed`                   | Random seed                         |
| `--use_cuda`               | Use GPU if available (flag)         |
| `--draft`                  | Enable draft/minimized mode (flag)  |
| `--topk`                   | Number of candidates                |
| `--learning_rate`          | Learning rate for encoder         |
| `--weight_decay`           | Weight decay for encoder          |
| `--train_batch_size`       | Batch size for training             |
| `--epoch`                  | Number of training epochs           |
| `--save_checkpoint_all`    | Save all checkpoints (flag)         |
| `--not_use_faiss`    | If set, means to use normal biosyn embeding (flag)         |



### Example:

```bash
python train.py \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --train_dictionary_path ./data/ncbi-disease/train_dictionary.txt \
    --train_dir ./data/ncbi-disease/processed_traindev \
    --output_dir ./data/output \
    --topk 20 \
    --epoch 10 \
    --train_batch_size 16\
    --learning_rate 1e-5 \
    --max_length 25 \
    --seed 0 \
```

```powershell
$ python train.py --model_name_or_path dmis-lab/biobert-base-cased-v1.1 --train_dictionary_path ./data/ncbi-disease/train_dictionary.txt --train_dir ./data/ncbi-disease/processed_traindev --output_dir ./data/output --topk 20 --epoch 10 --train_batch_size 16 --learning_rate 1e-5 --max_length 25 --seed 0
```

### Use FAISS:
The first advantage of this repo on the normal BioSyn, is introducing FAISS (Facebook AI Similarity Search) which allows us to do dense embedding comparison using FAISS indexes instead of np.argsort and np.matmul
You can control this by using the argument flag `--not_use_faiss`

