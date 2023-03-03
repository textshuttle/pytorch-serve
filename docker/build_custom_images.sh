#!/bin/sh

# build 0.7.1-cpu
./build_image.sh -bt dev -b  torchserve-0.7.1 -t textshuttle/pytorch-serve:0.7.1-cpu

# build 0.7.1-gpu
./build_image.sh -bt dev -g -cv cu113 -b torchserve-0.7.1 -t textshuttle/pytorch-serve:0.7.1-gpu

# build prio-cpu
./build_image.sh -bt dev -b feature/priority-linked-blocking-deque -t textshuttle/pytorch-serve:prio-cpu

# build prio-gpu
./build_image.sh -bt dev -g -cv cu113 -b feature/priority-linked-blocking-deque -t textshuttle/pytorch-serve:prio-gpu