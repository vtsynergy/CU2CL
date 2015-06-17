//This file is basically just a bootstrapper to ensure the examples all
// get translated both with CU2CL <= 0.6.2b and >= 0.7.0b

//Eventually, data validation will be added to ensure true semantic equivalence
// of these cases before they are moved to the "passing" directory

//This file also serves to test bugs relating to injecting code into the main
// method, in particular initialization and cleanup

//Assorted utilities for initializing and validating data
#include "../test_utils.hpp"

//Global configuration file, used by some of the test cases
#include "test_config.h"

//#include all the files, this will pull in some examples that do not have
// code that actually needs to be run (i.e. just examples demonstrating
// cases where output code needs to be cleaned, not checked for semantic
// equivalence
//Once CU2CL has a "generate error checking" option, many of the executed
// tests that do not need data validation (i.e. synthetic, nonfunctional
// kernels that just test syntax support) will have more value, as it will
// allow largely automatic diagnostics of problems with OpenCL generation
int count;
dim3 grid, block;

#include "kernel/const_qualifiers.cu"
#include "kernel/dynamic_smem.cu"
#include "kernel/extern_qualifiers.cu"
#include "kernel/integer_intrinsics.cu"
#include "kernel/macro_params.cu"
#include "kernel/pointer_qualifiers.cu"
#include "kernel/struct_params.cu"
#include "kernel/worksize.cu"

#include "memory/clmem_wrapper_funcs.cu"
#include "memory/host_alloc.cu"

#include "misc/macro_wrappers.cu"

int main(int argc, char ** argv) {

	//When adjusting the injection of the cleanup statement(s) ensure
	// it doesn't get added before these returns, only the final
	//(We do not support partial cleanup, and we cannot guarantee all
	// OpenCL state is ready for cleanup until the last statement in main)
	int error = 0;
	void * someData;
	if (error) return -1;
	if (cudaMalloc(&someData, 512) != cudaSuccess) error = 1;
	if (error) return -2;
	else cudaFree(someData);
	
	//Test errors relating to kernel code generation and invocation
		//Test support for translating dynamic shared memory
		testDynamicSmem();

		//Test support for handling different types of macros as params
		// to kernels as well as execution configuration
		testMacroParams();

		//Test support for appending OpenCL address space qualifiers
		// to various device-side pointers
		configPointerQualifiersTest(PQT_grid, PQT_block, PQT_count);
		testPointerQualifiers();

		//Test handling and accuracy of translating struct pointers
		// that are params to kernels (prone to alignment issues)
		configStructParamTest(TSP_grid, TSP_block, TSP_count);
		testStructParam();

		//Test construction of global and local worksize
		testWorksize();
	
	//Test errors relating to memory and address space translations
		//Test support for passing cl_mems through a wrapper function
		// separating the cudaMalloc from the kernel invocation
		testClmemWrapper();

		//Test dealing with cudaHostAlloc or cudaMallocHost memory
		configHostAlloc(GLOBAL_COUNT, 0);
		testHostAlloc1();
		testHostAlloc2();

	//Test any remaining unclassified errors
		//Test functions wrapped by a macro that translates to a function
		// a la the CUDA SDK's cutil.h
		configMacroTest(GLOBAL_COUNT);
		testMacroWrappedFuncs();

	//Make sure cleanup gets moved to right before this statement instead
	// of after, as it currently does
	return 0;
}
