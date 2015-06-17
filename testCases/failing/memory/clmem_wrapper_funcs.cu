//Test CU2CL's cl_mem generation, and how they are detected and forwarded
// through intermediate host-side functions between allocation and kernel

float * d_data;

__global__ void KernelFoo(float * data) {

}

void kernelWrapper(float * data) {
	dim3 grid(256, 1, 1);
	dim3 block(128, 1, 1);
	KernelFoo<<<grid, block>>>(data);
}

void testClmemWrapper() {
	cudaMalloc(&d_data, sizeof(float)*128);

	kernelWrapper(d_data);

	cudaFree(d_data);
}


