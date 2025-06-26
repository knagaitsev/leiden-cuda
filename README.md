## Leiden CUDA

**Simple setup to just run CUDA implementation:**

```bash
./tools/fetch_datasets.sh
cmake -S . -B build
cmake --build build
./build/csrc/leiden_test
```

This repository contains the following:

- Limited Leiden CUDA implementation - has refinement phase that works for small graphs, but focuses on greedy CPM node movement (`csrc/src/leiden_kernel.cu`)
- Fully working custom Leiden Python implementation (`custom_leiden.py`)
- Fully working custom Louvain Python implementation (`custom_louvain.py`)

## Running Python Code & Profiling

Advanced setup to run other tests:

```bash
conda create -y --prefix /path/to/dir python=3.12
conda activate /path/to/dir
conda install -y -c rapidsai -c conda-forge -c nvidia cugraph cuda-cudart cupy nx-cugraph cuda-version=12.0
conda install -y -c nvidia nsight-compute
pip install matplotlib pybind11 scipy pandas
```

Profile:

```bash
cmake --build build && ncu --set full --launch-skip 0 --target-processes all -o ./profiling/prof1 ./build/csrc/leiden_test
```

Additional setup if you want to try leidenalg (A different Python package):

```bash
pip install leidenalg igraph
```

You may need: https://igraph.org/c/#downloads

<!-- mkdir build && cd build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) ..
make -->

How to set up for adding Python bindings (WIP):

```bash
cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build
```
