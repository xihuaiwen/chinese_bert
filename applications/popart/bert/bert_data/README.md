# Graphcore benchmarks: BERT datasets

This directory contains the information required to create pre-training and training datasets for BERT.

`sample_text.txt` is a simple text file for initial pre-training with a small dataset.

The wikipedia dataset used for pre-training contains approximately 2.5 billion wordpiece tokens. This is only an approximate size since the wikipedia dump file is updated all the time.

SQuAD is a large reading comprehension dataset used for training (fine tuning) which contains 100,000+ question-answer pairs on 500+ articles. 

**NOTE**: these are large datasets - at least 300GB of disk space will be required and the data should be stored on NVMe SSDs for maximum performance. The examples below use a folder structure that matches the config files. If you use a different folder structure, make sure that it is correctly represented in the config file you use.

## File structure

The following files are found in the `bert_data/` folder:

* `create_pretraining_data.py`: creates pretraining dataset from a txt file
* `dataset.py`: dataset class
* `pretraining_dataset.py`: dataset class for pretraining task
* `squad_dataset.py`: dataset class for SQuAD fine tuning task
* `squad_utils.py`: utils function for SQuAD
* `sample_text.txt`:  sample text used as minimal example to pre-train BERT
* `wiki_downloader.sh`: downloads latest compressed xml file from wikipedia page
* `extract_wiki.sh`: parses wikipedia into articles
* `wikipedia_preprocessing.py`: outputs txt files that can be used as input for  `create_pretraining_data.py`
* `tokenization.py` Tokenizer file used to create pretraining data

## Pretraining data from `sample_text.txt`

The sample text is in a form that is already suitable to create the pretraining data. Run the script as:

`python3 create_pretraining_data.py --input-file path/to/sample_text.txt --output-file data/sample_text.bin --vocab-file path_to_the_vocab/vocab.txt --sequence-length 128 -- mask-tokens 20 --duplication-factor 6`

## Wikipedia pre-training data

All the instructions given below should be executed from the `bert_data/` folder. If necessary move to the  `bert_data/` folder first:

`cd /path-to-graphcore-bert/bert_data `

#### **1)** **Download the latest wikipedia dump**

Use the `wiki_downloader.sh` script to download the latest wikipedia dump:

`./wiki_downloader.sh path-to-target-folder-for-wikipedia-download-file`

This will download the latest wikipedia dump and place it into the folder passed as an argument.
It is then extracted into a `wikidump.xml` file that can be found inside the same folder.

#### **2)** **Clone the WikiExtractor git repository**

Clone the WikiExtractor git repository as follows:

`git clone https://github.com/attardi/wikiextractor.git`

**3)** **Extract the data**

Perform the extraction using `extract_wiki.sh` giving the path to the dump, the path to the destination folder, and the path to the WikiExtractor as arguments:

`./extract_wiki.sh /path/to/the/dump.xml /target_folder /path/to/Google's/WikiExtractor`

Inside the target folder there will be a directory called `AA/` that contains files named `wiki_00`, `wiki_01`...

These files have the following structure:

<doc id = article1>
Title of article 1

Body of article 1

</doc>

<doc id = article2>
Title of article 2

Body of article 2
</doc>

and so on.

If different filtering is required then use the WikiExtractor directly. A comprehensive list of options is shown here: https://github.com/attardi/wikiextractor.

**4)** **Preprocess the files**

The files from step 3 require further preprocessing with the `wikipedia_preprocessing.py` script:

`python3 wikipedia_preprocessing.py --input-file /target_folder/AA/ --output-file /preprocessed_target_folder`

where `target_folder/AA` contains the files from step 3 and `preprocessed_target_folder` will contain the new files (wiki_00_cleaned, wiki_01_cleaned, ...). The structure of the text in these files is now the same as the structure of the text in the `sample_text.txt` file.

**5) Tokenise the data**

The data can now be tokenised to create the pre-training dataset for BERT. For this step a vocabulary file is required. A vocabulary can be downloaded from the pre-trained model checkpoints at https://github.com/google-research/bert. We recommend to use the pre-trained BERT-Base Uncased model checkpoints. 

The script `create_pretraining_data.py` will accept a glob of input and output files to tokenise however attempting to process them all at once may result in the process being killed by the OS for consuming too much memory. It is therefore preferable to convert the files one by one:

`python3 create_pretraining_data.py --input-file /preprocessed_target_folder/wiki_00_cleaned --output-file /preprocessed_target_folder/wiki_00_tokenised --vocab-file path_to_the_vocab/vocab.txt --sequence-length 128 --mask-tokens 20 --duplication-factor 6`

**NOTE:** When using an uncased vocab, use `--do-lower-case`.

**NOTE:** Make sure to use the same values for `mask-tokens` and `duplication-factor` when generating the data and pretraining. 

The wikipedia dataset is now ready to be used in the Graphcore BERT model.

## Book-Corpus pre-training data

**1) Homemade BookCorpus**

Acquire the Homemade BookCorpus scripts from github and install their requirements:
```bash
git clone https://github.com/soskek/bookcorpus.git ~/bookcorpus
cd ~/bookcorpus
pip3 install -r requirements.txt
```

**2) URL list and File Download**
The following commands get the URL list and then download all the files in that list. For more details on these scripts consult the Homemade BookCorpus documentation.
```bash
python3 -u download_list.py > url_list.json
python3 download_files.py --list url_list.json --out files --trash-bad-count
```
On successful completeion you will now have approximately 15K documents in the directory .files/.

**3) Concatenate the files**
Concat all text to one sentence per line format using this script:
```bash
python3 make_sentlines.py files > all.txt
```

**4) Tokenise the data**
As with any other data (e.g. Wikipedia) the final step is to tokenise for both sequence lengths used in training:
```bash
cd path_to_gc_examples/applications/popart/bert/bert_data
python3 create_pretraining_data.py --input-file ~/bookcorpus/all.txt --output-file ~/bookcorpus/tokenised_128 --vocab-file path_to_the_vocab/vocab.txt --do-lower-case --sequence-length 128 --mask-tokens 20 --duplication-factor 6
python3 create_pretraining_data.py --input-file ~/bookcorpus/all.txt --output-file ~/bookcorpus/tokenised_384 --vocab-file path_to_the_vocab/vocab.txt --do-lower-case --sequence-length 384 --mask-tokens 60 --duplication-factor 6
```

The files ~/bookcorpus/tokenised_128 and ~/bookcorpus/tokenised_384 can now be used along-side the wikipedia data to pre-train the Graphcore BERT model.

## SQuAD training data

**1) Training files**

Get Google's SQuaD 1.1 training files as described in the README file here:

 https://github.com/google-research/bert

Use this command:

```bash
curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -o data/squad/train-v1.1.json
```

**2) Pre-trained weights**

Get Google's pre-trained weights (or produce your own by pre-training on the IPU). For example to get pre-trained weights for `BERT Base, uncased`:

`curl --create-dirs -L https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -o data/ckpts/uncased_L-12_H-768_A-12.zip`

Unzip the weights with:

`unzip data/ckpts/uncased_L-12_H-768_A-12.zip -d data/ckpts`

## SQuAD inference data

**1) SQuAD Inference files**

Get Google's SQuaD 1.1 inference files as described in the README file here:

 https://github.com/google-research/bert

Use this command:

```bash
curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/squad/dev-v1.1.json
curl -L https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py -o data/squad/evaluate-v1.1.py
```
