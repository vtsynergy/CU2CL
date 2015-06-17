//This test ensures CU2CL manages to preserve the struct keyword on
// translated kernel parameters
//Additionally, it serves to validate struct packing is preserved between
// host and device

//TODO: Add test of aligment handling, either here or in a separate case
#include <float.h>
#include <stdio.h>

struct TestStruct
{
	int intTest;
	short shortTest;
	float floatTest;
	char charTest;
};

//Take a packed struct and itemwise replace each floating point number
__global__ void
StructTestKernel(struct TestStruct * tests, float * replaceFloats, int n_elem) {
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	int threads = blockDim.x*gridDim.x;
	for (; tid < n_elem; tid += threads) {
		tests[tid].floatTest = replaceFloats[tid];
	}
}

void configStructParamTest(int itemCount, int gridx, int blockx) {
	count = itemCount;
	grid = gridx;
	block = blockx;
}

inline void initTestStructs(struct TestStruct * tests) {
	int i;
	for (i= 0; i < count; i++) {
		tests[i].intTest = i;
		tests[i].shortTest = (i & 0xffff);
		tests[i].floatTest = i * 1.0f;
		tests[i].charTest = (i & 0xff);
	}
} 

void testStructParam() {
	struct TestStruct * h_tests = (struct TestStruct *)malloc(sizeof(struct TestStruct)*count);
	struct TestStruct * d_tests;
	float * h_replace = (float *)malloc(sizeof(float)*count);
	float * d_replace;

	cudaMalloc(&d_tests, sizeof(struct TestStruct)*count);
	cudaMalloc(&d_replace, sizeof(float)*count);
	
	initTestStructs(h_tests);
	randomFillBuff<float>(h_replace, FLT_MIN, FLT_MAX, count);

	cudaMemcpy(d_tests, h_tests, sizeof(struct TestStruct)*count, cudaMemcpyHostToDevice);
	cudaMemcpy(d_replace, h_replace, sizeof(float)*count, cudaMemcpyHostToDevice);

	StructTestKernel<<<grid, block>>>(d_tests, d_replace, count);

	cudaMemcpy(h_tests, d_tests, sizeof(struct TestStruct)*count, cudaMemcpyDeviceToHost);

	//validate returned packed flaots
	int failures = 0, i = count-1;
	for(; i >= 0; i--)
		if (h_tests[i].floatTest != h_replace[i]) failures++;
	if (failures) fprintf(stderr, "testStructParam: Error: %d floatTests did not match intended replacement value!\n", failures);

	cudaFree(d_tests);
	cudaFree(d_replace);
	free(h_tests);
	free(h_replace);
}
