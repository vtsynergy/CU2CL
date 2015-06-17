//This file tests a few issues with tagging kernel pointer variables with
// their required address space
//It tests whether locally-declared pointers can be accurately qualified
// based on the qualifier added to any variables present in assignment
// statements on the locally-declared pointer
//It tests whether pointer params in __device__ or inline kernel functions
// can accurately inherit a __global__ parameter based on calls from a
// __global__ kernel function


__device__ void KernelDeviceReferenceTest(float& ref1, float& ref2) {
	ref1 = ref2 * 0.5f;
}

inline __device__ void KernelInlineDeviceReferenceTest(float& ref1, float& ref2) {
	ref1 = ref2 * 0.5f;
}

__device__ void KernelDevicePointerTest (float * ptr1, float * ptr2) {
	*ptr1 = *ptr2 * 0.5f;
}

inline __device__ void KernelInlineDevicePointerTest(float * ptr1, float * ptr2) {
	*ptr1 = *ptr2 * 0.5f;
}

__global__ void TestPointerQualifiers(float * arr1, float * arr2) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	float * ptr;

	//TODO add shared, local, and constant memory spaces

	ptr = &arr1[tid];

	KernelDeviceReferenceTest(arr1[tid], arr2[tid]);
	KernelInlineDeviceReferenceTest(arr2[tid], arr1[tid]);

	KernelDevicePointerTest(ptr, &arr2[tid]);
	KernelInlineDevicePointerTest(&arr2[tid], ptr);
}

void configPointerQualifiersTest(int gridSize, int blockSize, int itemCount) {
	count = itemCount;
	grid = gridSize;
	block = blockSize;
}

void testPointerQualifiers() {
	float * h_idata = (float *)malloc(sizeof(float)*count);
	float * h_odata = (float *)malloc(sizeof(float)*count);

	int i;
	for (i = 0; i < count; i++) h_idata[i] = 64.0f;

	float * d_data, * d_tempdata;

	cudaMalloc(&d_data, sizeof(float)*count);
	cudaMalloc(&d_tempdata, sizeof(float)*count);

	cudaMemcpy(d_data, h_idata, sizeof(float)*count, cudaMemcpyHostToDevice);

	TestPointerQualifiers<<<grid, block>>>(d_tempdata, d_data);

	cudaMemcpy(h_odata, d_data, sizeof(float)*count, cudaMemcpyDeviceToHost);

	//TODO validate the data

	cudaFree(d_data);
	cudaFree(d_tempdata);
	free(h_idata);
	free(h_odata);
}


