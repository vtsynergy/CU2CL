//This file tests wrapping an API function call in a macro that translates
// to an inline function

//Currently the macro wrapping breaks replacement semantics as it has trouble
// identifying the appropriate size to replace

//An ideal correct translation would translate the inline function fully, as
// well as approriately injecting the corresponding API call, without
// tampering with the macro wrapper

#include "../../test_utils.hpp"
#include <float.h>
#include <stdio.h>

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char *file, const int line) {
	fprintf(stderr, "%s:%i: CUDA Error: %s\n", cudaGetErrorString(err));
}

void configMacroTest(int testCount) {
	count = testCount;
}

void testMacroWrappedFuncs() {
	float * h_idata = (float*)malloc(count * sizeof(float));
	float * h_odata = (float*)malloc(count * sizeof(float));
	float * d_data;

	//init some host data
	randomFillBuff(h_idata, FLT_MIN, FLT_MAX, count); 

	safeCall(cudaMalloc(&d_data, count*sizeof(float)));
	safeCall(cudaMemcpy(d_data, h_idata, count*sizeof(float), cudaMemcpyHostToDevice));
	safeCall(cudaMemcpy(h_odata, d_data, count*sizeof(float), cudaMemcpyDeviceToHost));

	//check data integirty
	int failures;
	if(failures = checkBufBufExact(h_odata, h_idata, count)) fprintf(stderr, "testMacroWrappedFuncs: Error: %d float values did not match reference!\n", failures);

	safeCall(cudaFree(d_data));
	free(h_idata);
	free(h_odata);
}

