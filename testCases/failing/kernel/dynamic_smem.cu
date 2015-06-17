//This file tests CU2CL's ability to handle dynamically allocated shared memory
// (The third parameter to CUDA's execution configuration syntax)

//OpenCL's dynamic shared memory is actually more expressive than CUDA's since
// multiple distinct variables can be declared. In CUDA each pointer into smem
// space must be manually offset out of a single monolithic block

//An ideal translation adds a final extra kernel argument with the OpenCL
// __local address soace qualifier, removes all "extern shared" variables, and
// maps all __shared variables as offsets into the monolithic block
//To achieve this it must also move the 3rd parameter on the host side to be 
// the size of the final (new) clSetKernelArg, inheriting type appropriately)

__global__ void KernelDynamicSmemNoParams() {
	extern __shared__ float smem[];
	float * s_off = &smem[64];
}

__global__ void KernelDynamicSmemOneParam(float * param) {
	extern __shared__ float smem[];
	float * s_off = &smem[64];
}

__global__ void KernelDynamicSmem(float * param1, int param2) {
	extern __shared__ float smem[];
	float * s_off = &smem[64];
}


void testDynamicSmem() {
	float * d_param1;

	cudaMalloc(&d_param1, sizeof(float)*2048);

	//Test injecting a brand new clSetKernelArg (when none are present)
	KernelDynamicSmemNoParams<<<256, 64, 128*sizeof(float)>>>();

	//Test injecting a second clSetKernelArg (when only one is present)
	KernelDynamicSmemOneParam<<<256, 64, 512>>>(d_param1);

	//Test injecting a final clSetKernelArg (when multple are present)
	KernelDynamicSmem<<<256, 64, 128*sizeof(float)>>>(d_param1, 64);

	cudaFree(d_param1);
}
