#!/bin/bash
tool_chain_file=/home/xyoung/qnx/qnx700
. ${tool_chain_file}/qnxsdp-env.sh

mkdir build.qnx
cd build.qnx

TARGET_OS=qnx

cmake  -DCMAKE_TOOLCHAIN_FILE=${tool_chain_file}/qnx_cross_compile.cmake  ..
make -j6