#!/bin/bash

PATH_TO_COCO_DATASET=$1
PATH_TO_PARTITIONED_DATASET=$2
PATH_TO_MODEL=$3

if ! [[ -n "$PATH_TO_PARTITIONED_DATASET" ]]; then
    echo "Output path not set, setting to current directory."
    PATH_TO_PARTITIONED_DATASET="datasets"
fi

if [[ -n "$PATH_TO_MODEL" ]]; then
    echo "Model path set to $PATH_TO_MODEL"
    MODEL_PATH_ARGS="--model-path $PATH_TO_MODEL --pretrained-model-path $PATH_TO_MODEL"
fi

if [[ -n "$PATH_TO_COCO_DATASET" ]]; then
    cd "${0%/*}"

    echo "Creating and activating virtual environment"
    virtualenv -p python3 resnext_data_virtualenv
    source resnext_data_virtualenv/bin/activate    

    echo "Installing required python packages"
    pip3 install -r requirements.txt

    echo "Creating datasets and partitioning coco dataset"
    echo "Coco dataset: $PATH_TO_COCO_DATASET"
    for i in {1,2,8}
    do
        rm -r $PATH_TO_PARTITIONED_DATASET/dataset$i
        mkdir $PATH_TO_PARTITIONED_DATASET/dataset$i

        python3 partition_dataset.py --data-dir $PATH_TO_COCO_DATASET --partitions $i --output $PATH_TO_PARTITIONED_DATASET/dataset$i
        ret=$?
        if [ $ret -ne 0 ]; then
            exit $ret
        fi
    done

    echo "Getting pretrained model"
    for i in {1,2,3,4,5,6,8,10,12}
    do
        python3 get_model.py --micro-batch-size $i $MODEL_PATH_ARGS
        ret=$?
        if [ $ret -ne 0 ]; then
            exit $ret
        fi
    done

    echo "Deactivating virtual environment"
    deactivate
else
    echo "Must provide the path to the coco dataset as the first argument"
    exit 22
fi
