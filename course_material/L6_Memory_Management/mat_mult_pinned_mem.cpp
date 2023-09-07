/* Code to perform a Matrix multiplication using HIP
Written by Dr Toby M. Potter
*/

// Setup headers
#include <cassert>
#include <cmath>
#include <iostream>

// Bring in the size of the matrices
#include "mat_size.hpp"

// Bring in a library to manage matrices on the CPU
#include "mat_helper.hpp"

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

// standard matrix multiply kernel 
__global__ void mat_mult (
        float* A, 
        float* B, 
        float* C, 
        size_t N1_A, 
        size_t N0_C,
        size_t N1_C) { 
            
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)   
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We use row-major ordering for the matrices
    
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Scratch variable
    float temp=0.0f; 

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        // Get the offset within the memory allocation of C
        size_t offset = i0*N1_C+i1;
        
        // Loop over columns of A and rows of B
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A[i0*N1_A+n]*B[i1+n*N1_C]; 
        }
        
        // Set the value in C at offset
        C[offset]=temp;
        
        // Uncomment this to perform elementwise matrix multiplication instead
        // C[offset]=A[offset]*B[offset];
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

    //// Step 3. Construct matrices A_h and B_h on the host 
    //// and fill them with random numbers ////
    
    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(float);
    size_t nbytes_B = N1_A*N1_C*sizeof(float);
    size_t nbytes_C = N0_C*N1_C*sizeof(float);

    // Allocate pinned memory for the host arrays
    float *A_h, *B_h, *C_h;
    H_ERRCHK(
        hipHostMalloc(
            (void**)&A_h, 
            nbytes_A, 
            hipHostMallocDefault 
        )
    );
    H_ERRCHK(
        hipHostMalloc(
            (void**)&B_h, 
            nbytes_B,
            hipHostMallocDefault 
        )
    );
    
    // Allocate C_h in the normal way
    // But then register (pin) it!
    C_h = (float*)h_alloc(nbytes_C);
    
    // Register (pin) matrix C for use in the kernel
    H_ERRCHK(hipHostRegister(C_h, nbytes_C, hipHostRegisterDefault));

    // Fill the host arrays with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
    
    //// Step 4. Get device pointers for  //// 
    //// A_d, B_d, and C_d on the compute device ////

    float *A_d, *B_d, *C_d;
    H_ERRCHK(hipHostGetDevicePointer((void**)&A_d, A_h, 0));
    H_ERRCHK(hipHostGetDevicePointer((void**)&B_d, B_h, 0));
    H_ERRCHK(hipHostGetDevicePointer((void**)&C_d, C_h, 0));

    //// Step 5. Run the kernel to compute C_d ///
    //// from A_d and B_d on the device ////
        
    // Desired block size
    dim3 block_size = { 8, 8, 1 };
    dim3 global_size = { (uint32_t)N1_C, (uint32_t)N0_C, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=0;
    
    // Launch the kernel using hipLaunchKernelGGL method
    // Use 0 when choosing the default (null) stream
    hipLaunchKernelGGL(mat_mult, 
            grid_nblocks, 
            block_size, sharedMemBytes, 0, 
            A_d, B_d, C_d,
            N1_A,
            N0_C,
            N1_C
    );
    
    // Alternatively, launch the kernel using CUDA triple Chevron syntax
    // which is not valid ANSI C++ syntax
    //mat_mult<<<grid_nblocks, block_size, 0, 0>>>(A_d, B_d, C_d, N1_A, N0_C, N1_C);
    
    // Check for errors in the kernel launch
    H_ERRCHK(hipGetLastError());
    
    // Wait for any commands to complete on the compute device
    H_ERRCHK(hipDeviceSynchronize());
    
    //// Step 6. Test the computed matrix **C_h** against a known answer
    
    // Compute the serial solution using the matrix helper library
    float* C_answer_h = (float*)h_alloc(nbytes_C);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);
    

    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);
    
    //// Step 7. Write the contents of matrices A_h, B_h, and C_h to disk ////

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);
    h_write_binary(C_h, "array_C.dat", nbytes_C);
    
    //// Step 8. Clean up memory alllocations and release resources
    
    // Clean up pinned memory on the host   
    H_ERRCHK(hipHostFree(A_h));
    H_ERRCHK(hipHostFree(B_h));
    
    // Unregister host memory
    H_ERRCHK(hipHostUnregister(C_h));

    // Free the non-HIP host allocations in the normal way
    free(C_h);
    free(C_answer_h);
    
    // Reset compute devices
    h_reset_devices(num_devices);
}

