// standard matrix multiply kernel 
__kernel void mat_elementwise (
                        __global float* D, 
                        __global float* E,
                        __global float* F, 
                        unsigned int N0_F,
                        unsigned int N1_F) { 
            
    // F is of size (N0_F, N1_F)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 

    /// Insert missing kernel code ///
    /// To perform Hadamard matrix multiplication ///

} 