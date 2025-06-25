Simple setup:

```bash
./tools/fetch_datasets.sh
cmake -S . -B build
cmake --build build
./build/csrc/leiden_test
```

Advanced setup to run other tests:

```bash
conda create -y --prefix /pool/kir/conda/cuda-final-proj-2 python=3.12
conda activate /pool/kir/conda/cuda-final-proj-2
conda install -y -c rapidsai -c conda-forge -c nvidia cugraph cuda-cudart cupy nx-cugraph rmm cuda-version=12.0
conda install -y -c nvidia nsight-compute
pip install matplotlib pybind11 scipy pandas
```

Profile:

```bash
cmake --build build && sudo /pool/kir/conda/cuda-final-proj-2/bin/ncu --set full --launch-skip 0 --target-processes all -o ./profiling/prof1 ./build/csrc/leiden_test
```


<!-- mkdir build && cd build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) ..
make -->

How to set up for adding Python bindings:

```
cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build
```

Additional setup if you want to try leidenalg (A different Python package):

```bash
pip install leidenalg igraph
```

You may need: https://igraph.org/c/#downloads
