#!/bin/bash
DIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

##Directories for the distributed binary and compiled Clang install, shouldn't need to be changed unless the installer was modified
CU2CL_BIN_PATH=$DIR/cu2cl-build
CU2CL_CLANG_BIN_PATH="${DIR}/llvm-build/bin"

##Manually specify any extra include paths that must be forced
CU2CL_EXTRA_INCLUDES="-I /usr/include/x86_64-linux-gnu/c++/4.7"

##Launch clang with the CU2CL plugin
##extra compiler arguments, such as -I for additional include directories can be specified
`"${CU2CL_CLANG_BIN_PATH}"/clang -fsyntax-only -Xclang -load -Xclang "${CU2CL_BIN_PATH}"/RewriteCUDA.so -Xclang -plugin -Xclang rewrite-cuda -D __CUDACC__ -D __SM_35_INTRINSICS_H__ -D __SURFACE_INDIRECT_FUNCTIONS_H__ -D __SM_32_INTRINSICS_H__ -include "cuda_runtime.h" ${CU2CL_EXTRA_INCLUDES} -v "$@"`

