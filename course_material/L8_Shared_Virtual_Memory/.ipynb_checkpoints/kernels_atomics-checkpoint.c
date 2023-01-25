// Kernel to test atomics
__kernel void atomic_test (__global unsigned int* T) {
    
    // Increment T atomically
    atomic_int sum;
    atomic_init(&sum, 0);
    atomic_fetch_add(&sum, 1);
    
    T[0] = atomic_load(&sum);
}