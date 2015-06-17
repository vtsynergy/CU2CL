//A set of host-side utilities used by the testCases
// primarly methods to fill data structures with fixed or random data and 
// perform basic validation of data for checking semantic equivalence
#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP
template <typename T> void fillBuff(T* buffer, T value, int count);

template <typename T> void randomFillBuff(T* buffer, T min, T max, int count, int seed = 0);

template <typename T> int checkBufValExact(T* buffer, T val, int count);

template <typename T> int checkBufBufExact(T* buffer, T* reference, int count);

template <typename T> int checkBufValAbsError(T* buffer, T val, T tol, int count);

template <typename T> int checkBufBufAbsError(T* buffer, T* reference, T tol, int count);

template <typename T> int checkBufValRelError(T* buffer, T val, T tol, int count);

template <typename T> int checkBufBufRelError(T* buffer, T* reference, T tol, int count);

#endif //TEST_UTILS_HPP
