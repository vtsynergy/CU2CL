#!/bin/bash

##Download LLVM/CLang version 3.4 via tarball
echo "CU2CL: Fetching LLVM Tarball"
if [ ! -f llvm-3.4.src.tar.gz ]; then wget http://llvm.org/releases/3.4/llvm-3.4.src.tar.gz; fi
echo "CU2CL: Fetching Clang Tarball"
if [ ! -f clang-3.4.src.tar.gz ]; then wget http://llvm.org/releases/3.4/clang-3.4.src.tar.gz; fi
echo "CU2CL: Unzipping tarballs"
if [ ! -d llvm-3.4 ]; then tar -xzf llvm-3.4.src.tar.gz; fi
if [ ! -d llvm-3.4/tools/clang ]; then
tar -xzf clang-3.4.src.tar.gz
mv clang-3.4 llvm-3.4/tools/clang
fi

## Make Build Directories
if [ ! -d llvm-build ]; then mkdir llvm-build; fi

## Build LLVM and Clang -- REQUIRES CMAKE
cd llvm-build
echo "CU2CL: Configuring LLVM/Clang build"
cmake -DCMAKE_BUILD_TYPE=Release ../llvm-3.4
## adjust the argument to -j to suit available cores for speed
echo "CU2CL: Building LLVM/Clang"
make -j4

## Repair a few unlinked header files in this Clang Revision
echo "CU2CL: Repairing Clang headers"
if [ ! -L ../llvm-3.4/tools/clang/include/clang/Basic/AttrList.inc ]; then ln -s "$(pwd)"/tools/clang/include/clang/Basic/AttrList.inc ../llvm-3.4/tools/clang/include/clang/Basic/AttrList.inc; fi
if [ ! -L ../llvm-3.4/tools/clang/include/clang/Basic/DiagnosticCommonKinds.inc ]; then ln -s "$(pwd)"/tools/clang/include/clang/Basic/DiagnosticCommonKinds.inc ../llvm-3.4/tools/clang/include/clang/Basic/DiagnosticCommonKinds.inc; fi
if [ ! -L ../llvm-3.4/tools/clang/include/clang/AST/DeclNodes.inc ]; then ln -s "$(pwd)"/tools/clang/include/clang/AST/DeclNodes.inc ../llvm-3.4/tools/clang/include/clang/AST/DeclNodes.inc; fi
if [ ! -L ../llvm-3.4/tools/clang/include/clang/AST/StmtNodes.inc ]; then ln -s "$(pwd)"/tools/clang/include/clang/AST/StmtNodes.inc ../llvm-3.4/tools/clang/include/clang/AST/StmtNodes.inc; fi
if [ ! -L ../llvm-3.4/tools/clang/include/clang/AST/CommentCommandList.inc ]; then ln -s "$(pwd)"/tools/clang/include/clang/AST/CommentCommandList.inc ../llvm-3.4/tools/clang/include/clang/AST/CommentCommandList.inc; fi
if [ ! -L ../llvm-3.4/tools/clang/include/clang/AST/Attrs.inc ]; then ln -s "$(pwd)"/tools/clang/include/clang/AST/Attrs.inc ../llvm-3.4/tools/clang/include/clang/AST/Attrs.inc; fi
if [ ! -L ../llvm-3.4/tools/clang/include/clang/Driver/Options.inc ]; then ln -s "$(pwd)"/tools/clang/include/clang/Driver/Options.inc ../llvm-3.4/tools/clang/include/clang/Driver/Options.inc; fi

## Make a directory for the cu2cl shared library.
if [ ! -d ../cu2cl-build ]; then mkdir ../cu2cl-build; fi
cd ../cu2cl-build

## Use CMake to configure the build
echo "CU2CL: Configuring CU2CL build"
cmake -DCU2CL_PATH_TO_LLVM_SRC=../llvm-3.4 -DCU2CL_PATH_TO_CLANG_SRC=../llvm-3.4/tools/clang -DCU2CL_PATH_TO_LLVM_BUILD=../llvm-build -DCU2CL_PATH_TO_CLANG_BUILD=../llvm-build -DCMAKE_BUILD_TYPE=Release ../

## Then build CU2CL
echo "CU2CL: Building CU2CL"
make

## done
echo "CU2CL: Install complete!"
