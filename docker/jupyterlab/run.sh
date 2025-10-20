#!/bin/bash

# Note the forward slash before `$(pwd)` to ensure correct volume mounting on Windows
docker run \
    -p 8888:8888 \
    -v /$(pwd)/../../data:/workspace/data \
    -v /$(pwd)/../../notebooks:/workspace/notebooks \
    -it jupyterlab
