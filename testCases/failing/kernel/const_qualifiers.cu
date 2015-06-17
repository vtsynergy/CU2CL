//This case tests CU2CL's ability to remove the const qualifier from host
// variables when they are pulled out of kernel files

const int constInt1 = 10;
int const constInt2 = 20;

__global__ void KernelConstQualTest() {

	const int constInt3 = 30;
	int const constInt4 = 40;

}
