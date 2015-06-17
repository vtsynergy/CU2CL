//Test CU2CL's coverage of the kernel math API's integer intrinsics

//All should have an appropriate translation with at least as much accuracy
// or else an appropriate error emitted

__global__ void KernelTestIntIntrinsics() {
	int x = 256, y = 256;
	long long int llx = 256, lly = 256;
	unsigned long long int ullx = 256, ully = 256;
	unsigned int ux = 256, uy = 256, us = 256;

	__brev(ux);
	__brevll(ullx);
	__byte_perm(ux, uy, us);
	__clz(x);
	__clzll(llx);
	__ffs(x);
	__ffsll(llx);
	__hadd(x, y);
	__mul24(x ,y);
	__mul64hi(llx, lly);
	__mulhi(x, y);
	__popc(ux);
	__popcll(ullx);
	__rhadd(x, y);
	__sad(x, y, ux);
	__uhadd(ux, uy);
	__umul24(ux, uy);
	__umul64hi(ullx, ully);
	__umulhi(ux, uy);
	__urhadd(ux, uy);
	__usad(ux, uy, us);
}
