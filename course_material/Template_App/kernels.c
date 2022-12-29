// standard matrix multiply kernel 
__kernel void template (__global float* C,  
                        unsigned int N0_C,
                        unsigned int N1_C) { 
            
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    float temp=0.0; 

    // Guard mechanism
    if ((i0<N0_C) && (i1<N1_C)) {
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
} 