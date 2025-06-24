
Setup:

```bash
conda create -y --prefix /pool/kir/conda/cuda-final-proj-2 python=3.12
conda activate /pool/kir/conda/cuda-final-proj-2
conda install -y -c rapidsai -c conda-forge -c nvidia cugraph cuda-cudart cupy nx-cugraph rmm cuda-version=12.0
pip install matplotlib pybind11 scipy pandas
```

<!-- mkdir build && cd build
cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) ..
make -->

```
cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build
```

Additional setup if you want to try leidenalg (A different Python package):

```bash
pip install leidenalg igraph
```

https://igraph.org/c/#downloads
