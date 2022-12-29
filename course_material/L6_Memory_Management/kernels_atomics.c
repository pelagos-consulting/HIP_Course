// Kernel to test atomics
__kernel void atomics_test1 (__global unsigned int* T) {
    
    // Increment T atomically
    atomic_add(T, 1);
}
