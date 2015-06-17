//This case is used to demonstrate a case where CU2CL
// can generate a multiple declaration of anew cl_mem variable
// that results when translating a cudaHostAlloc'd pointer

//This occurs as the variable is generated at global scope, rather than
// inheriting the scope of the original CUDA pointer

//A correct translation would move the global declaration of __cu2cl_Mem_h_odata
// into the two individual functions where h_odata is locally-declared
#include "../../test_utils.hpp"

int combined;

void configHostAlloc(int testCount, int writeCombined) {
	count = testCount;
	combined = writeCombined;
}

//Test host-allocing two buffers, and transfering data through a device buffer
void testHostAlloc1() {
	unsigned char * h_idata = NULL;
	unsigned char * h_odata = NULL;
	unsigned char * d_idata;
#if CUDART_VERSION >= 2020
	cudaHostAlloc((void**)&h_idata, count*sizeof(unsigned char), combined ? cudaHostAllocWriteCombined : 0);
	cudaHostAlloc((void**)&h_odata, count*sizeof(unsigned char), combined ? cudaHostAllocWriteCombined : 0);
#else
	cudaMallocHost((void**)&h_idata, count*sizeof(unsigned char));
	cudaMallocHost((void**)&h_odata, count*sizeof(unsigned char));
#endif

	//init some host data in h_idata
	randomFillBuff<unsigned char>(h_idata, 0, 255, count);

	cudaMalloc((void**) &d_idata, count*sizeof(unsigned char));

	cudaMemcpy(d_idata, h_idata, count*sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaMemcpy(h_odata, d_idata, count*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	//check data integrity
	checkBufBufExact<unsigned char>(h_odata, h_idata, count);

	cudaFreeHost(h_idata);
	cudaFreeHost(h_odata);
	cudaFree(d_idata);
}

void testHostAlloc2() {
	unsigned char * h_odata = NULL;
	unsigned char * d_idata;
#if CUDART_VERSION >= 2020
	cudaHostAlloc((void**)&h_odata, count*sizeof(unsigned char), combined ? cudaHostAllocWriteCombined : 0);
#else
	cudaMallocHost((void**)&h_odata, count*sizeof(unsigned char));
#endif

	//No need to init data, the validity is established by testHostAlloc1
	cudaMalloc((void**)&d_idata, count*sizeof(unsigned char));
	cudaMemcpy(d_idata, h_odata, count*sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaFreeHost(h_odata);
	cudaFree(d_idata);
}
