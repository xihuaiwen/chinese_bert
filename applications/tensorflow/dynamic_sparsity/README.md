# Graphcore: Dynamic Sparsity

---
## Training with Dynamic Sparse Matrices

The Poplar SDK (since version 1.2) supports dynamic element wise
sparse weight matrices where the sparsity pattern can be
dynamically changed at runtime. The examples here use
custom TensorFlow ops to access this dynamic sparsity
support from the Poplibs/popsparse library.

### File structure

* ipu_sparse_ops/ A Tensorflow Python module for accessing IPU sparse custom ops.
* mnist_rigl/ Train MNIST using the ipu_sparse_ops module to implement a dymanic sparse optimiser (Rig-L)
* `README.md` This file.

### How to use this application

1) Prepare the environment.

   Install the `poplar-sdk` (version 1.2 or later) following the README provided. Make sure to source the `enable.sh`
   scripts for poplar and gc_drivers.

2) Install required modules:

```bash
pip install -r requirements.txt
```

3) Build the custom ops and then run the python code. The command below runs a simple test of the ipu_sparse_ops module. To use this module it must be on your `PYTHONPATH`. (Below the environment variable is set only for this single command but you can append ipu_sparse_ops permanently to your python path if you intend to use the module regularly).

Build the ipu_sparse_ops module and test it:
```bash
make -j
PYTHONPATH=./ python ipu_sparse_ops/tools/sparse_matmul.py
```

To run any of the sparse models/training see the READMEs in their respective folders e.g. mnist_rigl/README.md
