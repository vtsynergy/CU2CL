//A set of host-side utilities used by the testCases
// primarly methods to fill data structures with fixed or random data and 
// perform basic validation of data for checking semantic equivalence
#include <stdlib.h>
#include <time.h>
#include <cmath>

template <typename T> void fillBuff(T* buffer, T value, int count) {
	int i = count-1;
	for(; i >= 0; i--)
		buffer[i] = value;
}

template <typename T> void randomFillBuff(T* buffer, T min, T max, int count, int seed = 0) {
	if (seed == 0)
		srand(time(NULL));
	else
		srand(seed);

	int i = count-1;
	for(; i >= 0; i--) 
		buffer[i] = (rand() * (max-min))/RAND_MAX + min;
}

template <typename T> int checkBufValExact(T* buffer, T val, int count) {
	int failures = 0, i = count-1;
	for (; i >=0; i--)
		if (buffer[i] != val) failures++;
	return failures;
}

template <typename T> int checkBufBufExact(T* buffer, T* reference, int count) {
	int failures = 0, i = count-1;
	for (; i >= 0; i--)
		if (buffer[i] != reference[i]) failures++;
	return failures;
}

template <typename T> int checkBufValAbsError(T* buffer, T val, T tol, int count) {
	int failures = 0, i = count-1;
	for (; i >= 0; i--)
		if (abs(buffer[i]-val) > tol) failures++;
	return failures;
}

template <typename T> int checkBufBufAbsError(T* buffer, T* reference, T tol, int count) {
	int failures = 0, i = count-1;
	for (; i >= 0; i--)
		if (abs(buffer[i]-reference[i]) > tol) failures++;
	return failures;
}

template <typename T> int checkBufValRelError(T* buffer, T val, T tol, int count) {
	int failures = 0, i = count-1;
	for (; i >= 0; i--)
		if (abs((buffer[i]-val)/buffer[i]) > tol) failures++;
	return failures;
}

template <typename T> int checkBufBufRelError(T* buffer, T* reference, T tol, int count) {
	int failures = 0, i = count-1;
	for (; i >= 0; i--)
		if (abs((buffer[i]-reference[i])/buffer[i]) > tol) failures++;
	return failures;
}

template void randomFillBuff<float>(float*, float, float, int, int);
template void randomFillBuff<unsigned char>(unsigned char*, unsigned char, unsigned char, int, int);

template int checkBufBufExact<float>(float*, float*, int);
template int  checkBufBufExact<unsigned char>(unsigned char*, unsigned char *, int);
