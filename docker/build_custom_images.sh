#!/bin/sh

# build cpu
./build_image.sh -t textshuttle/pytorch-serve:torchserve-23mt-v0.8.0-v2-cpu

# build gpu
./build_image.sh -g -cv cu118 -t textshuttle/pytorch-serve:torchserve-23mt-v0.8.0-v2-gpu