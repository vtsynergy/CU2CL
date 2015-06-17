//This test checks for CU2CL's ability to remove the  extern "C"  qualifier
// from forward function declarations when removing them from device code
//It also tests extern variables and the block extern statement


extern "C" void fooFunc();

extern "C" int fooInt;

extern "C" {
	void blockFunc();
	int blockInt;
}

__global__ void KernelFoo() {

}
