#!/bin/sh

# build 23mt-cpu
./build_image.sh -bt dev -b torchserve-23mt -t textshuttle/pytorch-serve:23mt-cpu

# build 23mt-gpu
./build_image.sh -bt dev -g -cv cu113 -b torchserve-23mt -t textshuttle/pytorch-serve:23mt-gpu
