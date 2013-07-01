project(CU2CL)
cmake_minimum_required(VERSION 2.8)

set(CU2CL_PATH_TO_LLVM_SRC "" CACHE PATH
    "Path to LLVM source code. Not necessary if using an installed LLVM.")
set(CU2CL_PATH_TO_LLVM_BUILD "" CACHE PATH
    "Path to the directory where LLVM was built or installed.")

set(CU2CL_PATH_TO_CLANG_SRC "" CACHE PATH
    "Path to Clang source code. Not necessary if using an installed Clang.")
set(CU2CL_PATH_TO_CLANG_BUILD "" CACHE PATH
    "Path to the directory where Clang was built or installed.")

if(CU2CL_PATH_TO_LLVM_SRC)
    if(NOT EXISTS "${CU2CL_PATH_TO_LLVM_SRC}/cmake/config-ix.cmake")
        message(FATAL_ERROR
                "Please set CU2CL_PATH_TO_LLVM_SRC to the root directory of LLVM source code.")
    else()
        get_filename_component(LLVM_SRC_DIR ${CU2CL_PATH_TO_LLVM_SRC}
                               ABSOLUTE)
        list(APPEND CMAKE_MODULE_PATH "${LLVM_SRC_DIR}/cmake/modules")
    endif()
endif()
if(CU2CL_PATH_TO_CLANG_SRC)
    if(NOT EXISTS "${CU2CL_PATH_TO_CLANG_SRC}/CMakeLists.txt")
        message(FATAL_ERROR
            "Please set CU2CL_PATH_TO_CLANG_SRC to the root directory of Clang source code.")
    else()
        get_filename_component(CLANG_SRC_DIR ${CU2CL_PATH_TO_CLANG_SRC}
                               ABSOLUTE)
    endif()
endif()

#if installing from combined LLVM/Clang SVN source tree, use /bin/llvm-tblgen
#otherwise, use /bin/tblgen
if(NOT EXISTS "${CU2CL_PATH_TO_LLVM_BUILD}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}" )
    message(FATAL_ERROR
            "Please set CU2CL_PATH_TO_LLVM_BUILD to a directory containing an LLVM build.")
endif()
get_filename_component(LLVM_BUILD_DIR ${CU2CL_PATH_TO_LLVM_BUILD}
                       ABSOLUTE)
if(NOT EXISTS "${CU2CL_PATH_TO_CLANG_BUILD}/bin/clang-3.2${CMAKE_EXECUTABLE_SUFFIX}" )
    message(FATAL_ERROR
            "Please set CU2CL_PATH_TO_CLANG_BUILD to a directory containing a Clang build.")
endif()
get_filename_component(CLANG_BUILD_DIR ${CU2CL_PATH_TO_CLANG_BUILD}
                       ABSOLUTE)

list(APPEND CMAKE_MODULE_PATH "${LLVM_BUILD_DIR}/share/llvm/cmake")

include(AddLLVM)
include(HandleLLVMOptions)
#if building from combined LLVM/Clang SVN source tree, use /share/llvm/cmkae/LLVMConfig.cmake
#otherwise use LLVM.cmake
include("${LLVM_BUILD_DIR}/share/llvm/cmake/LLVMConfig.cmake")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories("${LLVM_BUILD_DIR}/include" "${LLVM_SRC_DIR}/include")
include_directories("${CLANG_BUILD_DIR}/include" "${CLANG_SRC_DIR}/include")
link_directories("${LLVM_BUILD_DIR}/lib")
link_directories("${CLANG_BUILD_DIR}/lib")

llvm_process_sources(srcs RewriteCUDA.cpp)

add_library(RewriteCUDA MODULE ${srcs})

set(LLVM_USED_LIBS
    clangFrontend
    clangRewrite
    clangAST
    clangLex
    clangBasic
)

foreach(lib ${LLVM_USED_LIBS})
    target_link_libraries(RewriteCUDA ${lib})
endforeach(lib)

set_target_properties(RewriteCUDA
    PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX ""
)
