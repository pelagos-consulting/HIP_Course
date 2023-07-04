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
    // Number of work-items along a dimension of workgroup
    size_t local_length,
    // Number of items in the array
    size_t array_length,
    // Index of work item along dimension of workgroup
    size_t local_index,
    // Starting position of the copy
    size_t *start,
    // End position of the copy
    size_t *end) {
  
    // Work out the jump size
    size_t jump_size=array_length/local_length;
    if (array_length%local_length) jump_size++;
    
    // Starting position for the copy
    *start=local_index*jump_size;
    // End position for the copy
    *end=(local_index+1)*jump_size;
    // Limit end so we don't go off the end
    *end=min(*end,array_length);
}    

// Matrix multiply kernel that uses local memory for B
__kernel void mat_mult_local (
                        __global float* A, 
                        __global float* B, 
                        __global float* C,
                        __local  float* shared_B,
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C) { 
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // shared_B is of size (L1, N1_A)
    // C is of size (N0_C, N1_C)
    
    // Make a local scratch array for demonstration purposes
    // (not actually used)
    __local float scratch[10];
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Location within the workgroup
    size_t s0=get_local_id(1);
    size_t s1=get_local_id(0);
    
    // Local size
    size_t L0=get_local_size(1);
    size_t L1=get_local_size(0);
    
    // start and end
    size_t start, end;
    
    // Fill shared_B
    
    // Get the start and end lengths
    get_start_end(L0, N1_A, s0, &start, &end);
    // Fill the columns of shared with B
    if (i1<N1_C) {
        for (size_t n=start; n<end; n++) {
            shared_B[s1*N1_A+n]=B[i1+n*N1_C]; 
        }
    }
    
    // Enqueue a local barrier to make sure all the work items finish
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
            // shared_B is of size (L1, N1_A)
            // C is of size (N0_C, N1_C)
            
            // Loop across row i0 of A
            // and across row s1 of shared_B
            temp+=A[i0*N1_A+n]*shared_B[s1*N1_A+n]; 
        } 
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1]=temp;
    }
}

// Matrix multiply kernel that uses local memory for B
__kernel void mat_mult_local_vector (
                        __global float8* A, 
                        __global float* B, 
                        __global float* C,
                        __local  float8* shared_B,
                        unsigned int N1_A, 
                        unsigned int N0_C,
                        unsigned int N1_C,
                        // N1_A_v is the number of float8 vectors along N1_A axis
                        // We have asserted that N1_A%8 == 0
                        unsigned int N1_A_v) { 
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // shared_B is of size (L1, N1_A)
    // C is of size (N0_C, N1_C)
    
    // Make a local scratch array for demonstration purposes
    // (not actually used)
    __local float scratch[10];
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0=get_global_id(1); 
    size_t i1=get_global_id(0); 
    
    // Location within the workgroup
    size_t s0=get_local_id(1);
    size_t s1=get_local_id(0);
    
    // Local size
    size_t L0=get_local_size(1);
    size_t L1=get_local_size(0);
    
    // start and end
    size_t start, end;
    
    // Fill shared_B
    
    // Temporary scratch vector
    float8 temp=(float8)0.0f;
    
    // Get the start and end lengths
    get_start_end(L0, N1_A_v, s0, &start, &end);
    // Fill the columns of shared with B
    if (i1<N1_C) {
        for (size_t n=start; n<end; n++) {
            // Bn is the starting location in B
            __global float* Bn=&B[i1+n*8*N1_C];
            // SBn is the starting location for shared_B
            __local float8* SBn=&shared_B[s1*N1_A_v+n];
            // Fill individual components of the vector            
            temp.s0=Bn[0*N1_C];
            temp.s1=Bn[1*N1_C];
            temp.s2=Bn[2*N1_C];
            temp.s3=Bn[3*N1_C];
            temp.s4=Bn[4*N1_C];
            temp.s5=Bn[5*N1_C];
            temp.s6=Bn[6*N1_C];
            temp.s7=Bn[7*N1_C];
            
            // Insert the temporary variable into place
            *SBn=temp;          
        }
    }
    
    // Enqueue a local barrier to make sure all the work items finish
    barrier(CLK_LOCAL_MEM_FENCE);

    temp=(float8)0.0f;
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A_v; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            // shared_B is of size (L1, N1_A)
            // C is of size (N0_C, N1_C)
            
            // Loop across row i0 of A
            // and across row s1 of shared_B
            temp+=A[i0*N1_A_v+n]*shared_B[s1*N1_A_v+n];
        } 
        // Use vector indexing to collapse the vector
        C[i0*N1_C+i1]=temp.s0+temp.s1+temp.s2+temp.s3+
            temp.s4+temp.s5+temp.s6+temp.s7;
    }
}

