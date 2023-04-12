// Kernel to test OpenCL atomics v2.0
__kernel void atomics_test2 (__global atomic_uint* T) {
      
    // Increment T atomically using OpenCL 2.0
    atomic_fetch_add(T, 1);
}
