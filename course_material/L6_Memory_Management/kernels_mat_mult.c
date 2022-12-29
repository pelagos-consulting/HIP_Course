// Example of a global variable
#ifdef __opencl_c_program_scope_global_variables
__global int a_g = 2; 
__global float b_g[2] = {2.0,1.0}; 
#endif

// Example of constant memory
__constant float pi = 3.1415;
__constant float coeffs[] = {1.0, -2.0, 1.0};

// Kernel function to get the start and end values
// for filling a shared memory array
void get_start_end(
    size_t local_length, 
    size_t array_length,
    size_t local_index,
    size_t *start,
    size_t *end) {
  
    // Work out the jumps
    size_t jump=array_length/local_length;
    if (array_length%local_length) jump++;
    *start=local_index*jump;
    *end=(local_index+1)*jump;
    *end=min(*end,array_length);
}    

// Matrix multiply kernel that uses local memory
__kernel void mat_mult_local (
                        __global float* A, 
                        __global float* B, 
                        __global float* C,
                        __local  float* shared_A,
                        __local  float* shared_B,
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)
    
    // Make a local scratch array for demonstration purposes
    // (not actually used)
    __local float scratch[10];
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    // Location within the workgroup
    size_t s0=get_local_id(0);
    size_t s1=get_local_id(1);
    
    // Local size
    size_t L0=get_local_size(0);
    size_t L1=get_local_size(1);
    
    // start and end
    size_t start0, end0, start1, end1;
    
    // Fill shared memory
    
    // Get the start1 and end1 lengths to fill a block
    get_start_end(L1, N1_A, s1, &start1, &end1);
    // Fill shared_A with the rows of A
    if (i0<N0_C) {
        for (size_t n=start1; n<end1; n++) {
            shared_A[s0*N1_A+n]=A[i0*N1_A+n]; 
        }
    }   
    
    // Get the start0 and end0 lengths
    get_start_end(L0, N1_A, s0, &start0, &end0);
    // Fill the columns of shared with B
    if (i1<N1_C) {
        for (size_t n=start0; n<end0; n++) {
            shared_B[s1*N1_A+n]=B[n*N1_C+i1]; 
        }
    }
    
    // Enqueue a local barrier to make sure shared memory is filled
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Scratch variable whose allocation uses constant memory pi
    float temp=0.0*pi; 
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            // C is of size (N0_C, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=shared_A[s0*N1_A+n]*shared_B[s1*N1_A+n]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
}

// Local memory matrix multiply kernel
// where B has been transposed and using vectors
__kernel void mat_mult_local_transp_vec (
                        __global float8* A_star, 
                        __global float8* BT_star, 
                        __global float* C,
                        __local  float8* shared_A_star,
                        __local  float8* shared_BT_star,
                        unsigned int N1_A_v, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A_star is of size (N0_C, N1_A_v)
    // BT_star is of size (N1_C, N1_A_v)
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1);
    
    //printf("%zu %zu\n", i0, i1);
    
    // Location within the workgroup
    size_t s0=get_local_id(0);
    size_t s1=get_local_id(1);
    
    // Size of the workgroup
    size_t L0=get_local_size(0);
    size_t L1=get_local_size(1);
    
    // start and end
    size_t start0, end0, start1, end1;
    
    // Get the start1 and end1 lengths to fill a block
    get_start_end(L1, N1_A_v, s1, &start1, &end1);
    // Fill shared_A with the rows of A
    if (i0<N0_C) {
        for (size_t n=start1; n<end1; n++) {
            shared_A_star[s0*N1_A_v+n]=A_star[i0*N1_A_v+n]; 
        }
    }   
    
    // Get the start0 and end0 lengths
    get_start_end(L0, N1_A_v, s0, &start0, &end0);
    // Fill the rows of shared_BT_star with BT_star
    if (i1<N1_C) {
        for (size_t n=start0; n<end0; n++) {
            shared_BT_star[s1*N1_A_v+n]=BT_star[i1*N1_A_v+n]; 
        }
    }
    
    // Enqueue a local barrier to make sure shared memory is filled
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Scratch variable whose allocation uses constant memory pi
    float8 temp=(float8)(0.0f); 
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of shared_A_star 
        // and columns of BT_star 
        for (size_t n=0; n<N1_A_v; n++) {
            
            // Local size
            // shared_A_star is of size (L0, N1_A_v)
            // shared_BT_star is of size (L1, N1_A_v)    
            // C is of size (N0_C, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=shared_A_star[s0*N1_A_v+n]*shared_BT_star[s1*N1_A_v+n]; 
        } 
        
        // Number of rows in C is same as number of rows in A
        
        // sum over the elements of the vector
        C[i0*N1_C+i1]=(
            temp.s0+temp.s1+temp.s2+temp.s3
            +temp.s4+temp.s5+temp.s6+temp.s7
        );
    }
}

// Local memory matrix multiply kernel 
__kernel void transpose (__global float* src, 
                        __global float* dest, 
                        unsigned int N0_src,
                        unsigned int N1_src) { 
    
    // src is of size (N0_src, N1_src)
    // dest is of size (N1_src, N0_src)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(0); 
    size_t i1=get_global_id(1); 
    
    if ((i0<N0_src) && (i1<N1_src)) {
        dest[i1*N0_src+i0]=src[i0*N1_src+i1];
    }
}
