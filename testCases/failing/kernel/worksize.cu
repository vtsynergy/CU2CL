//This test check's CU2CL's ability to handle generating localWorkSize and
// globalWorkSize appropriately in a few cases where something other than a
// dim3 is provided to the execution configuration syntax

#define THREADS 128

__global__ void KernelNoSmem() {

}

__global__ void KernelWithSmem() {
 extern __shared__ float foo[];

}

void testWorksize() {

	const int threads = THREADS;
	int N = 1024;
	dim3 grid(8, threads, 1);
	KernelNoSmem<<<grid, THREADS>>>();

	
	KernelWithSmem<<<N, N / threads, N*sizeof(float)>>>();

	
	KernelNoSmem<<<8, 128>>>();

}
