# Graphcore

---
## Advanced Poplar Example

This example performs a simple computation but uses more advanced
features of Poplar than the SDK tutorials. This includes:

* Moving I/O into separate Poplar programs
* Use of multiple programs
* Use of Poplibs
* Saving and restoring of Poplar executables
* Skipping graph and program construction when restoring executables
* Enabling Profiling
* Choosing the number of IPUs
* Choosing between IPU HW and IPU Model for execution

### File structure

* `main.cpp` The main Poplar code example.
* `Makefile` A simple Makefile for building the example.
* `utils.h` Utility functions used by the example.
* `codelets.cpp` A custom vertex used in the example.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh`
    scripts for poplar, gc_drivers (if running on hardware).

   The example also uses boost::program_options. You can install boost via your package manager.

2) Build and run the example.

```
make
./example --help
```
