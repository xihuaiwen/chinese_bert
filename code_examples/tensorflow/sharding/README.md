# Graphcore
---
## Sharding examples

### File structure

* `simple_sharding.py` Minimal example of IPU sharding
* `README.md` This file.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the poplar-sdk following the README provided. Make sure to run the enable.sh scripts and activate a Python virtualenv with gc_tensorflow installed.

2) Run the script.

        python simple_sharding.py [--autoshard]

    The `--autoshard` flag enables automatic sharding across 2 IPUs