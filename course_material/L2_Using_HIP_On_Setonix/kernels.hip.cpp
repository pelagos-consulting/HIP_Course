// Kernel files must include the hip runtime
#include <hip/hip_runtime.h>

// Kernel to fill a vector 
__global__ void fill (float* A, float fill_value, size_t N) { 
            
    // A is of size (N,)
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i0<N) {
        A[i0]=fill_value;
    }
} 
