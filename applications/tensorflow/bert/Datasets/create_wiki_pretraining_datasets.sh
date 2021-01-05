#!/bin/bash

# This script will create the Wikiepdia dataset used to pretraing BERT. 
# It will download the latest Wikipedia dump archive and preprocess, tokenize, and convert to Tensorflow TFRecord files.
# The entire process may take up to several hours.

export LANG=C.UTF-8
usage() {
        echo -e "\nUsage:\n$0 Download_data_path\n"
}

if [  $# -ne 1 ]
then
    usage
    exit 1
fi

download_path="$1/wikidata"
dump_path="${download_path}/AA"
preprocess_path="${download_path}/preprocess"


if [[ ! -f "vocab.txt" ]]; then
    echo 'File "vocab.txt" does not exist, please download from an existing source, for example  "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt".'
fi


if [ ! -d $download_path ];then
    mkdir $download_path
    echo "Dateset loading......"
    bash wiki_download.sh $download_path
fi

# Install wikiextractor package to extract wiki dump file. 
echo "Install wikiextractor......"
pip install wikiextractor

# Extractor and split wikidump.xml to `wiki_00`,`wiki_01`, ...
if [ ! -d $dump_path ];then
    echo "Data unziping......"
    bash wiki_extract.sh $download_path'/wikidump.xml' $download_path
fi
# Further preprocessing to `wiki_00_cleaned`,`wiki_00_cleaned`, ... 
if [ ! -d $preprocess_path ];then
    mkdir $preprocess_path
    echo "Data preprocesing......"
    python wiki_preprocess.py --input-file $dump_path --output-file $preprocess_path
fi

echo "Generating datasets with sequence length 128......"

bash create_pretraining_tfrecord.sh "${download_path}" 128 20 5

echo "Generating datasets with sequence length 384......"

bash create_pretraining_tfrecord.sh "${download_path}" 384 58 5