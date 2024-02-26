#!/bin/bash
# tool_chain_file=/home/xyoung/qnx/qnx700
# . ${tool_chain_file}/qnxsdp-env.sh

mkdir build
cd build

TARGET_OS=qnx

cmake    ..
# make -j6