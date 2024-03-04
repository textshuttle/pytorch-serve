#!/bin/sh

# increment when appropriate
VERSION=2

# build cpu
./build_image.sh -bt dev -b torchserve-23mt-v0.8.0 -t textshuttle/pytorch-serve:torchserve-23mt-v0.8.0-v${VERSION}-cpu

# build gpu
./build_image.sh -bt dev -b torchserve-23mt-v0.8.0 -g -cv cu118 -t textshuttle/pytorch-serve:torchserve-23mt-v0.8.0-v${VERSION}-gpu

# note that this will build arm/amd images depending on your OS
# for multi-platform building and tagging, see https://github.com/textshuttle/pytorch-serve/pull/15#issuecomment-1906360733
