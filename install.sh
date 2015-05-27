#!/bin/bash

## Download LLVM/CLang revision 159674 -- REQUIRES SVN
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

## Repair a few unlinked header files in this Clang Revision
ln -s `pwd`/tools/clang/include/clang/Basic/AttrList.inc ../llvm/tools/clang/include/clang/Basic/AttrList.inc
ln -s `pwd`/tools/clang/include/clang/Basic/DiagnosticCommonKinds.inc ../llvm/tools/clang/include/clang/Basic/DiagnosticCommonKinds.inc
ln -s `pwd`/tools/clang/include/clang/AST/DeclNodes.inc ../llvm/tools/clang/include/clang/AST/DeclNodes.inc
ln -s `pwd`/tools/clang/include/clang/AST/StmtNodes.inc ../llvm/tools/clang/include/clang/AST/StmtNodes.inc
ln -s `pwd`/tools/clang/include/clang/AST/Attrs.inc ../llvm/tools/clang/include/clang/AST/Attrs.inc

## Make a directory for the cu2cl shared library.
mkdir ../cu2cl-build
cd ../cu2cl-build

## Use CMake to configure the build
cmake -DCU2CL_PATH_TO_LLVM_SRC=../llvm -DCU2CL_PATH_TO_CLANG_SRC=../llvm/tools/clang -DCU2CL_PATH_TO_LLVM_BUILD=../llvm-build -DCU2CL_PATH_TO_CLANG_BUILD=../llvm-build -DCMAKE_BUILD_TYPE=Release ../

## Then build CU2CL
make

## done
