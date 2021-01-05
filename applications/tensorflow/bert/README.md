# Bert Training on IPUs
This repository provides a script and recipe to run BERT models for NLP pre-training and training on Graphcore IPUs.

# BERT models
BERT Base requires 16 IPUs for pre-training (Wikipedia). Better performance can be achieved by putting 2 hidden layers in the same IPU then doing the replicas(i.e. set `replicas` to 2). 

**NOTE**: IPUs can only be acquired in powers of two (2, 4, 8, 16). Unused IPUs will be unavailable for other tasks.

## Datasets
The wikipedia dataset contains approximately 2.5 billion wordpiece tokens. This is only an approximate size since the wikipedia dump file is updated all the time.

At least 1TB of disk space will be required for full pre-training (two phases, phase 1 with sequence_length=128 and phase 2 with sequence_length=384) and the data should be stored on NVMe SSDs for maximum performance.
If full pre-training is required (with the two phases with different sequence lengths) then data will need to be generated separately for the two phases:
- once with --sequence-length 128 --mask-tokens 20 --duplication-factor 5
- once with --sequence-length 384 --mask-tokens 58 --duplication-factor 5

See the `Datasets/README.md` file for more details on how to generate this data.


## Runnint the models

|    File    |       Description         |
|------------|---------------------------|
| `run_pretraining.py`      | Main training loop of pre-training task |
| `ipu_utils.py`  | IPU specific utilities |
| `ipu_optimizer.py`  | IPU optimizer |
| `log.py`        | Module containing functions for logging results |
| `multi_stage_wrapper.py` | Wrapper for splitting embedding lookup to multiple stages
| `Datasets/`     | Code for using different datasets.<br/>-`data_loader.py`: Dataloader and preprocessing.<br/>-`create_pretraining_data.py`: Script to generate tfrecord files to be loaded from text data|
| `modeling.py`       | A Pipeline Model description for pretrain task on the IPU.
| `LR_Schedules/` | Different LR schedules<br/> - `exponential.py`: A exponential learning rate schedule with optional warmup.<br/>-`natural_exponential.py`: A natural exponential learning rate schedule with optional warmup.<br/>-`custom.py`: A customized learning rate schedule with given `lr_schedule_by_step`.


## Quick start guide

### Prepare environment
**1) Download the Poplar SDK**

[Download](https://downloads.graphcore.ai/) and install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh` scripts for poplar, gc_drivers (if running on hardware) and popART.

**2) python**

Create a virtural environment and install the required packages:
```shell
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
pip install <path to gc_tensorflow.whl>
```

### Generate pretraining data(small sample)
As an example we will create data from a small sample: `Datasets/sample.txt`, however the steps are the same for a large corpus of text. As described above, see `Datasets/README.md` for instructions on how to generate data for the Wikipedia and SQuAD datasets.

**Download the vocab file**

You can download a vocab from the pre-trained model checkpoints at https://github.com/google-research/bert. For this example we are using `Bert-Base, uncased`.

**Creating the data**

Create a directory to keep the data.
```shell
mkdir data
```
`Datasets/create_pretraining_data.py` has a few options that can be viewed by running with `-h/--help`.
Data for the sample text is created by running:
```shell
python3 Datasets/create_pretraining_data.py \
  --input-file Datasets/sample.txt \
  --output-file Datasets/sample.tfrecord \
  --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt \
  --do-lower-case \
  --sequence-length 128 \
  --mask-tokens 20 \
  --duplication-factor 10 \
  --remask
```
**NOTE**: `--input-file/--output-file` can take multiple arguments if you want to split your dataset between files.
When creating data for your own dataset, make sure the text has been preprocessed as specified at https://github.com/google-research/bert. This means with one sentence per line and documents delimited by empty lines.

### Pre-training with BERT on IPU
For the sample text a configuration has been created - `configs/demo.json`. It sets the following options:
```
{
    "task": "pretraining",
    "attention_probs_dropout_prob": 0.0,
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "max_position_embeddings": 512,
    # Two layers as our dataset does not need the capacity of the usual 12 Layer BERT Base
    "num_hidden_layers": 2,
    "hidden_layers_per_stage": 1,
    "vocab_size": 30400,
    "seq_length": 128,
    "batch_size": 1,
    # The data generation should have created 64 samples. Therefore, we will do an epoch per session.run
    "batches_per_step": 64,
    "epochs": 100,
    "max_predictions_per_seq": 20,
    "base_learning_rate": 0.0004,
    "lr_schedule": "exponential",

    "precision": "16",
    "seed": 1234,
    "loss_scaling": 1.0,
    "optimiser": "momentum",
    "momentum": 0.9,
    # The pipeline depth should be at least twice the number of pipeline stages
    "pipeline_depth": 16,
    "pipeline_schedule": "Grouped",
    "replicas": 1,
    "do_train":true,
    # Here we specify the file we created in the previous step
    "train_file": "Datasets/sample.tfrecord"
  }
```
Run this config:
```shell
python3 run_pretraining.py --config configs/demo.json
```

**View the pre-training results in Tensorboard**
`requirements.txt` will install a standalone version of tensorboard. The program will log all training runs to `--log-dir`(`checkpoints` by default). View them by running:
```shell
tensorboard --logdir=checkpoints
```
#### Run the training loop for pre-training (Wikipedia)
For BERT Base phase 1, use the following command:
```shell
python3 run_pretraining.py --config configs/pretrain_base_128.json
```
For BERT Base phase 2, use the following command:
```shell
python3 run_pretraining.py --config configs/pretrain_base_384.json
```
**Note**: Don't forget to set `init_checkpoint` to checkpoint path where phase 1 saved when training phase 2. If you want to use Google's pretrained checkpoint to run Graphcore BERT phase 2, you can find a simple script in `ipu_utils.py`. To do that, try with:
```python3
from ipu_utils import convert_google_ckpt_to_gc
convert_google_ckpt_to_gc(google_ckpt, output_dir='./')
```
Then you'll get the coverted checkpoint files in `gc_ckpt` folder.


### Training options
`run_pretraining.py` has many different options. Run with `-h/--help` to view them. Any options used on the command line will overwrite those specified in the configuration file.

Currently we can reach **80.49/87.97** EM/F1 score at **3** epoches and **81.07/88.16** at **4** epoches with SQuAD task, throughtputs get about **1540 samples/s** and time-to-train(TTT) is about **37 hours**, with following **phase 1** options:
```
  "attention_probs_dropout_prob": 0.0,
  "hidden_layers_per_stage": 2,
  "batch_size": 1,
  "batches_per_step": 102,
  "epochs": 1,
  "base_learning_rate": 0.0001,
  "lr_schedule": "natural_exponential",
  "loss_scaling": 20.0,
  "optimiser": "momentum",
  "momentum": 0.98,
  "pipeline_depth": 108,
  "replicas": 2
```

And the **phase 2** options are:
```
  "attention_probs_dropout_prob": 0.0,
  "hidden_layers_per_stage": 1,
  "batch_size": 1,
  "batches_per_step": 46,
  "epochs": 1,
  "base_learning_rate": 0.00001,
  "lr_schedule": "natural_exponential",
  "loss_scaling": 20.0,
  "optimiser": "momentum",
  "momentum": 0.984375,
  "pipeline_depth": 180,
  "replicas": 1
```

More options are set by default in `configs/pretrain_base_128.json` and `configs/pretrain_base_384.json`.

## SQuAD with BERT on IPU
To run on SQuAD the necessary files can be found here:
https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

Download these to some directory $SQUAD_DIR.
Modify SQuAD config file `configs/squad_base.json` like:
'''
  "vocab_file":"$SQUAD_DIR/vocab.txt",
  "train_file":"$SQUAD_DIR/train-v1.1.json",
  "predict_file":"$SQUAD_DIR/dev-v1.1.json",
'''

Config `init_checkpoint` options:
'''
  "init_checkpoint":"$PRETRAIN_DIR/ckpt-name-trans/ckpt-trans",
'''

For SQuAD fine-tuning, 
  config `do_training` options for train:
  '''
    "do_training": true,
    "do_predict": false,
  '''
  run command:
  ```shell
  python3 run_squad.py --config configs/squad_base.json
  ```

  Suppose fine-tuning checkpoint stored in $FINETUNE_DIR,config `do_predict` options for predict:
  '''
    "do_training": false,
    "do_predict": true,
    "init_checkpoint":"$FINETUNE_DIR/ckpt-name",
  '''
  run command:
  ```shell
  python3 run_squad.py --config configs/squad_base.json
  ```

After SQuAD predict, the dev set predictions will be saved into a file called `predictions.json` in the "tfrecord_dir"

For SQuAD evaluating, use the following command:
```shell
python3 evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $tfrecord_dir/predictions.json
```

