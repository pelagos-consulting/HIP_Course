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
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_F) && (i1<N1_F)) {
        
        // Create an offset
        size_t offset = i0*N1_F+i1;
        
        // Number of rows in C is same as number of rows in A
        F[offset]=D[offset]*E[offset];
    }
} 
