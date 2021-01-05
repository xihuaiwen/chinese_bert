#!/bin/bash
# This script will create BERT datasets for pretraining task. 
# It will create a folder named like `tokenised_seq128_mask20_dup2`, 
# and preprocess, tokenize, and convert the input files to Tensorflow TFRecord files
# The entire process may take up to several hours.
# TODO: This script take single file as input,  process and duplicate several
# TODO: times within this file, then save as TFRecord file. This may cause insufficient
# TODO: shuffle. However we reached SOTA with datasets generated by this scipt. So it
# TODO: should work but we might need other script to make data shuffling more sufficient.

export LANG=C.UTF-8
usage() {
        echo -e "\nUsage:\n$0 wiki_root_path sequence_length num_mask_tokens duplication_factor\n"
}

if [  $# -ne 4 ]
then
    usage
    exit 1
fi
wiki_root_path=${1}
sequence_length=${2}
num_mask_tokens=${3}
duplication_factor=${4}

preprocessed_wiki_path=$wiki_root_path/preprocess
if [ ! -d "$preprocessed_wiki_path" ]; then
        echo "Folder $preprocessed_wiki_path does not exist. Please generate the cleaned raw datasets first."
        exit 1
fi

output_path=$wiki_root_path/'tokenised_seq'$sequence_length'_mask'$num_mask_tokens'_dup'$duplication_factor
if [ ! -d "$output_path" ]; then
        mkdir -p "$output_path"
fi
# parallel processing.
create_wiki_tfrecord(){
        for i in $(seq -f "%02g" $1 $2);do
                name='wiki_'$i'_cleaned'
                echo $name' processing......'
                output=$output_path/$name.tfrecord
                echo 'Save to '$output
                logfile='process_'${name}'_seq'${sequence_length}.log
                nohup python3 create_pretraining_data.py \
                --input-file $preprocessed_wiki_path/$name \
                --output-file $output \
                --vocab-file ./vocab.txt \
                --sequence-length $sequence_length \
                --mask-tokens $num_mask_tokens \
                --duplication-factor $duplication_factor \
                --do-lower-case \
                --remask 2>&1 > ${logfile} &
        done
}
# Manully split to 2 parts, each parts process 7 files in parallel.
# It can change to more parts if there is no enough memory.
create_wiki_tfrecord 00 06
wait

echo "First part finished. Start to second part."

create_wiki_tfrecord 07 13
wait


