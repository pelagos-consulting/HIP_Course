/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

//// Step 1. Setup headers and parse command line arguments ////

#include <cassert>
#include <cmath>
#include <iostream>

// Bring in the size of the matrices
#include "mat_size.hpp"

// Bring in the library to work with matrices
#include "mat_helper.hpp"

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

typedef float float_type;
typedef float4 float_vec_type;

// Device function to get the start and end values
// for filling a shared memory array
__device__ void get_start_end(
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

// Matrix multiply kernel that uses shared memory for A
__global__ void mat_mult_shared_A_vector (
                        float_vec_type* A, 
                        float_type* B, 
                        float_type* C,
                        size_t vector_len,
                        size_t N1_A_v,
                        size_t N0_C,
                        size_t N1_C) { 
    
    // Access the allocation of shared memory
    extern __shared__ char shared[];
    
    // Get a pointer to shared_A from shared
    float_vec_type* shared_A = (float_vec_type*)&shared[0];
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // shared_A is of size (L0, N1_A)
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Location within the workgroup
    size_t s0=threadIdx.y;
    size_t s1=threadIdx.x;
    
    // Local size
    size_t L0=blockDim.y;
    size_t L1=blockDim.x;
    
    // start and end positions for the copy
    size_t start, end;
    
    // Get the start and end lengths to fill array
    get_start_end(L1, N1_A_v, s1, &start, &end);
    
    // Fill shared_A
    if (i0<N0_C) {
        for (size_t n=start; n<end; n++) {
            shared_A[s0*N1_A_v+n]=A[i0*N1_A_v+n]; 
        }   
    }
    
    // Set a barrier to ensure that all threads 
    // sync to this point before moving on 
    __syncthreads();
    
    // Scratch variables
    float_vec_type temp = (float_vec_type){0.0f}; 
    float_vec_type scratch = (float_vec_type){0.0f};
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A_v; n++) {
            
            // A is of size (N0_C, N1_A_v)
            // B is of size (N1_A, N1_C)
            // shared_A is of size (L0, N1_A_v)
            // C is of size (N0_C, N1_C)
             
            float_type* Bn_i1=&B[n*vector_len*N1_C+i1];
            
            // Fill components of scratch
            scratch.x = Bn_i1[0*N1_C];
            scratch.y = Bn_i1[1*N1_C];
            scratch.z = Bn_i1[2*N1_C];
            scratch.w = Bn_i1[3*N1_C];
            
            // Perform the dot product using shared memory
#ifdef __HIP_PLATFORM_NVIDIA__
            temp.x += shared_A[s0*N1_A_v+n].x*scratch.x;
            temp.y += shared_A[s0*N1_A_v+n].y*scratch.y;
            temp.z += shared_A[s0*N1_A_v+n].z*scratch.z;
            temp.w += shared_A[s0*N1_A_v+n].w*scratch.w;
#else
            temp+=shared_A[s0*N1_A_v+n]*scratch;
#endif
        } 
        
        // Number of rows in C is same as number of rows in A
        C[i0*N1_C+i1] = temp.x+temp.y+temp.z+temp.w;
    }
}

int main(int argc, char** argv) {

    //// Step 1. Parse program arguments ////

    // Parse command line arguments
    int dev_index = h_parse_args(argc, argv);
    
    // Number of devices discovered
    int num_devices=0;
    
    //// Step 2. Discover resources and choose a compute device ////
    
    // Helper function to acquire devices
    // This sets the default device
    h_acquire_devices(&num_devices, dev_index);
        
    // Report on the device in use
    h_report_on_device(dev_index);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)    
    // C is of size (N0_C, N1_C)

    size_t N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;
    
    // Get the vector length
    size_t vector_len = sizeof(float_vec_type)/sizeof(float_type);

    // Get the nearest acceptable vector length
    size_t N1_A_v = N1_A/vector_len;
    if (N1_A%vector_len) N1_A_v++;
    
    //// Step 3. Construct matrices A_h and B_h on the host 
    //// and fill them with random numbers ////
    
    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(float_type);
    size_t nbytes_B = N1_A*N1_C*sizeof(float_type);
    size_t nbytes_C = N0_C*N1_C*sizeof(float_type);

    // Allocate memory for the host arrays
    float* A_h = (float_type*)h_alloc(nbytes_A);
    float* B_h = (float_type*)h_alloc(nbytes_B);
    float* C_h = (float_type*)h_alloc(nbytes_C);

    // Fill the host arrays with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
    
    //// Step 4. Allocate memory for arrays //// 
    //// A_d, B_d, and C_d on the compute device ////

    
    // Number of bytes in enlarged array
    size_t nbytes_A_v = N0_C*N1_A_v*sizeof(float_vec_type);
    float_vec_type *A_d; 
    float_type *B_d, *C_d;
    
    // Allocate memory for A_d, B_d and C_d normally
    H_ERRCHK(hipMalloc((void**)&A_d, nbytes_A_v));
    H_ERRCHK(hipMalloc((void**)&B_d, nbytes_B));
    H_ERRCHK(hipMalloc((void**)&C_d, nbytes_C));

    // Fill A_d with zeros
    H_ERRCHK(
        hipMemset(
            A_d, // the pointer to set
            0, // the value to fill
            nbytes_A_v // number of bytes to fill
        )
    );
    
    //// Step 5. 1. Upload matrices A_h and B_h from the host //// 
    //// to A_d and B_d on the device ////
    
    // Copy a rectangular region into A
    
    // Memcpy2D method
    H_ERRCHK(
        hipMemcpy2D(
            (void*)A_d, // destination pointer
            N1_A_v*sizeof(float_vec_type), // destination pitch
            (void*)A_h, // source pointer
            N1_A*sizeof(float_type), // source pitch
            N1_A*sizeof(float_type), // width of pencils to copy
            N0_C, // number of pencils to copy
            hipMemcpyHostToDevice // type of memory transfer
        )
    );
        
    // Copy B normally using a contiguous copy
    H_ERRCHK(hipMemcpy(B_d, B_h, nbytes_B, hipMemcpyHostToDevice));
 
    //// Step 6. Run the kernel to compute C_d ///
    //// from A_d and B_d on the device ////
        
    // Desired block size
    dim3 block_size = { 8, 8, 1 };
    dim3 global_size = { (uint32_t)N1_C, (uint32_t)N0_C, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=block_size.y*N1_A_v*sizeof(float_vec_type);
    
    // Launch the kernel using hipLaunchKernelGGL method
    // Use 0 when choosing the default (null) stream
    hipLaunchKernelGGL(mat_mult_shared_A_vector, 
            grid_nblocks, 
            block_size, sharedMemBytes, 0, 
            A_d, B_d, C_d,
            vector_len,
            N1_A_v,
            N0_C,
            N1_C
    );
    
    // Alternatively, launch the kernel using CUDA triple Chevron syntax
    // which is not valid ANSI C++ syntax
    //mat_mult<<<grid_nblocks, block_size, sharedMemBytes, 0>>>(A_d, B_d, C_d, N1_A, N0_C, N1_C);
    
    // Check for errors in the kernel launch
    H_ERRCHK(hipGetLastError());
    
    // Wait for any commands to complete on the compute device
    H_ERRCHK(hipDeviceSynchronize());

    //// Step 7. Copy the buffer for matrix C_d //// 
    //// on the device back to C_h on the host ////
    H_ERRCHK(hipMemcpy((void*)C_h, (const void*)C_d, nbytes_C, hipMemcpyDeviceToHost));
    
    //// Step 8. Test the computed matrix **C_h** against a known answer
    
    // Compute the serial solution using the matrix helper library
    float* C_answer_h = (float*)calloc(nbytes_C, 1);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);
    
    // Uncomment this to check against elementwise matrix multiplication
    // m_hadamard(A_h, B_h, C_answer_h, N0_C, N1_C);

    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);
    
    //// Step 9. Write the contents of matrices 
    ///  A_h, B_h, and C_h to disk 

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);
    h_write_binary(C_h, "array_C.dat", nbytes_C);
    
    //// Step 10. Clean up memory alllocations and release resources
    
    // Free the HIP buffers
    H_ERRCHK(hipFree(A_d));
    H_ERRCHK(hipFree(B_d));
    H_ERRCHK(hipFree(C_d));

    // Clean up host memory
    free(A_h);
    free(B_h);
    free(C_h);

    // Free the answer matrix
    free(C_answer_h);
    
    // Reset compute devices
    h_reset_devices(num_devices);
}

