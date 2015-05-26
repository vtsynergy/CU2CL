#!/bin/bash

##Download LLVM/CLang revision 159674 -- REQUIRES SVN
svn checkout -r 159674 http://llvm.org/svn/llvm-project/llvm/trunk llvm
cd llvm/tools
svn checkout -r 159674 http://llvm.org/svn/llvm-project/cfe/trunk clang
cd ../..

## Make Build Directories
mkdir llvm-build

## Build LLVM and Clang -- REQUIRES CMAKE
cd llvm-build
cmake -DCMAKE_BUILD_TYPE=Release ../llvm
## adjust the argument to -j to suit available cores for speed
make -j2

cd ..
#done
