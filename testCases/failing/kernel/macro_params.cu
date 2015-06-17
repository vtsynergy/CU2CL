//This test is intended to test CU2CL's ability to accurately transform
// kernel parameters that involve macros
//It tests both macro constants and function-like macros

#define TEST1 256
#define TEST2 TEST1
#define MIN(a,b) ((a<b) ? a : b)
#define MINTEST(a) MIN(a, TEST2)


__global__ void MacroParamKernel(int p1, int p2, int p3, int p4, int p5, int p6) {
	//TODO do something
}

void testMacroParams(){

	MacroParamKernel<<<TEST1, MIN(TEST1, 256)>>>(TEST1, TEST2, 512 + TEST2, MIN(256, 512), MIN(TEST2, 512), MINTEST(512));
}
