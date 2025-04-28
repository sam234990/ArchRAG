# C-HNSW

we implement the C-HNSW with the [faiss framework](https://github.com/facebookresearch/faiss) (V1.8.0), and called it HCHNSW in the source code (e.g., IndexHCHNSW).
For install and run code, please follow these steps:


## Building from source summary

```shell
# step 1
# for Debug
$ cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Debug
# for release
$ cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON 

# step 2
$ make -C build -j faiss

# step 3 Building the python bindings (optional)
$ make -C build -j swigfaiss
$ (cd build/faiss/python && python setup.py install)

# step 4 C++ library and headers
$ make -C build install

# step 5 test
# for C++
$ make -C build test 
# for Python
$ (cd build/faiss/python && python setup.py build)
$ PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py

#step demo debug
$ make -C build 6-2-HCHNSW
$ ./build/tutorial/cpp/6-2-HCHNSW
$ ./build/demos/demo_ivfpq_indexing
```

### Tips

When running Python scripts that depend on Faiss or other compiled libraries, you may encounter the following error:

```bash
ImportError: /path/to/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by .../_swigfaiss.so)
```

This happens because the installed version of `libstdc++.so.6` (GNU C++ standard library) is too old and does not include the required `GLIBCXX_3.4.32` symbol.

The easiest fix is to install an updated version of gcc in your Conda environment, which will provide a compatible `libstdc++.so.6`:

```bash
$ conda install -c conda-forge gcc
```