Graphcore: CosmoFlow example
===
This README describes how to run a conv3D based model called CosmoFlow on IPU hardware with synthetic data.  

## File structure

* `configs` 	     folder containing the config .yaml file
* `data_gen`		     folder containing code for input data pipeline
* `models`   	     folder containing code defining graph models
* `README.md`	     The file you are reading now
* `requirements.txt`   Python dependencies
* `train.py`	     Main python script to launch and run the example
* `utils`	             folder containing utility functions
* `LICENSE`	     license document
* `test_cosmoflow.py`  unit test for CosmoFlow

## How to use this code example

1. Prepare the IPU environment.

 - Download, install & activate the Poplar SDK following instructions provided in the Getting Started document
	
 - Create python virtual environment, install gc_tensorflow and active the Python environment. This is also documented in the Getting Started guide. This example uses TensorFlow v1, so please ensure your Python venv has TFv1 installed. 


2. Start running the example with one of the following commands:

 - run without tensorflow estimator, with 1 IPU:
   `python train.py configs/graphcore.yaml`

 - run without tensorflow estimator, with 2 IPUs:
   `python train.py configs/graphcore.yaml --num-ipus=2`

 - run with tensorflow estimator, with 1 IPU:
   `python train.py configs/graphcore.yaml --use-estimator`

 - run with tensorflow estimator, with 2 IPUs:
   `python train.py configs/graphcore.yaml --use-estimator --num-ipus=2`


## License

The code for this example is based on code published at https://github.com/sparticlesteve/cosmoflow-benchmark/
Its license is retained and reproduced here as LICENSE. 

models/resnet.py was derived by the original author from https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py.
The original license for that code is preserved at the start of the file, and is compatible with the license for the modifications, and the rest of this example. 
