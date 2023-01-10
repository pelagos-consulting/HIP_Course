/* Code to perform a Matrix multiplication using HIP
Written by Dr Toby M. Potter
*/

// Setup headers
#include <cassert>
#include <cmath>
#include <iostream>

// Bring in the size of the matrices
#include "mat_size.hpp"

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

// standard matrix multiply kernel 
__global__ void mat_mult (
        float* A_d, 
        float* B_d, 
        float* C_d, 
        size_t N1_A, 
        size_t N0_C,
        size_t N1_C) { 
            
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Scratch variable
    float temp=0.0; 

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        // Loop over columns of A and rows of B 
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A_d[i0*N1_A+n]*B_d[n*N1_C+i1]; 
        } 
        // Number of rows in C is same as number of rows in A
        C_d[i0*N1_C+i1]=temp;
    }
} 

int main(int argc, char** argv) {
    
    //// Step 1. Parse program arguments ////

    // Parse arguments and set the target device
    int dev_index = h_parse_args(argc, argv);
    
    // Number of devices discovered
    int num_devices;
    
    //// Step 2. Discover resources ////
    
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
    size_t nbytes_A, nbytes_B, nbytes_C;

    //// Step 3. Read matrices A and B from file into host memory ////
    
    float* A_h = (float*)h_read_binary("array_A.dat", &nbytes_A);
    float* B_h = (float*)h_read_binary("array_B.dat", &nbytes_B);

    // Sanity check on incoming data
    assert(nbytes_A==N0_C*N1_A*sizeof(float));   
    assert(nbytes_B==N1_A*N1_C*sizeof(float));
    nbytes_C=N0_C*N1_C*sizeof(float);
    
    // Make an array on the host to store the result (matrix C)
    float* C_h = (float*)calloc(nbytes_C, 1);
    
    //// Step 4. Allocate on-device memory for matrices A, B, and C ////

    float *A_d, *B_d, *C_d;
    H_ERRCHK(hipMalloc((void**)&A_d, nbytes_A));
    H_ERRCHK(hipMalloc((void**)&B_d, nbytes_B));
    H_ERRCHK(hipMalloc((void**)&C_d, nbytes_C));

    //// Step 5. Upload matrices A and B from the host

    // to the HIP device allocations ////
    H_ERRCHK(hipMemcpy(A_d, A_h, nbytes_A, hipMemcpyHostToDevice));
    H_ERRCHK(hipMemcpy(B_d, B_h, nbytes_B, hipMemcpyHostToDevice));
 
    //// Step 6. Run the kernel to compute C from A and B ////
        
    // Desired block size
    dim3 block_size = { 8, 8, 1 };
    dim3 global_size = { (uint32_t)N1_C, (uint32_t)N0_C, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);
    
    // Run the kernel
    hipLaunchKernelGGL(mat_mult, 
            grid_nblocks, 
            block_size, 0, 0, 
            A_d, B_d, C_d,
            N1_A,
            N0_C,
            N1_C
    );
    
    // Wait for any commands to complete on the compute device
    H_ERRCHK(hipDeviceSynchronize());

    //// Step 7. Copy the Buffer for matrix C back to the host ////
    H_ERRCHK(hipMemcpy(C_h, C_d, nbytes_C, hipMemcpyDeviceToHost));
    
    //// Step 8. Write the contents of matrix C to disk
    
    // Write out the result to file
    h_write_binary(C_h, "array_C.dat", nbytes_C);

    //// Step 9. Clean up arrays and release resources
    
    // Free the HIP buffers
    H_ERRCHK(hipFree(A_d));
    H_ERRCHK(hipFree(B_d));
    H_ERRCHK(hipFree(C_d));

    // Clean up memory that was allocated on the read   
    free(A_h);
    free(B_h);
    free(C_h);
    
    // Clean up devices
    h_release_devices(num_devices);
}

