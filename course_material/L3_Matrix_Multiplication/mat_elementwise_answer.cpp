/* Code to perform Hadamard (elementwise) multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

// Define the size of the arrays to be computed
#define NROWS_F 520
#define NCOLS_F 1032

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

// standard matrix multiply kernel 
__global__ void mat_elementwise (
    float* D, 
    float* E,
    float* F, 
    size_t N0_F,
    size_t N1_F) { 
            
    // F is of size (N0_F, N1_F)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;

    //// Insert missing kernel code ////
    /// To perform Hadamard matrix multiplication ///
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_F) && (i1<N1_F)) {
        
        // Create an offset
        size_t offset = i0*N1_F+i1;
        
        // Number of rows in C is same as number of rows in A
        F[offset]=D[offset]*E[offset];
    }

    //// End insert code ////
} 

int main(int argc, char** argv) {
    
    // Parse arguments and set the target device
    int dev_index = h_parse_args(argc, argv);
    
    // Number of devices discovered
    int num_devices;
    
    //// Step 2. Discover resources ////
    
    // Helper function to acquire devices
    // This sets the default context 
    h_acquire_devices(&num_devices, dev_index);
        
    // Report on the device in use
    h_report_on_device(dev_index);
        
    // D, E, F is of size (N0_F, N1_F)
    size_t N0_F = NROWS_F, N1_F = NCOLS_F;
    size_t nbytes_D, nbytes_E, nbytes_F;

    // Read the input data into arrays on the host and sanity check
    float* D_h = (float*)h_read_binary("array_D.dat", &nbytes_D);
    float* E_h = (float*)h_read_binary("array_E.dat", &nbytes_E);

    // Sanity check on incoming data
    assert(nbytes_D==N0_F*N1_F*sizeof(float));   
    assert(nbytes_E==N0_F*N1_F*sizeof(float));
    nbytes_F=N0_F*N1_F*sizeof(float);
    
    // Make an array to store the result in array_F
    float* F_h = (float*)calloc(nbytes_F, 1);
    
    // Allocate memory on device for arrays D, E, and F
    float *D_d, *E_d, *F_d;
    
    h_errchk(hipMalloc((void**)&D_d, nbytes_D));
    h_errchk(hipMalloc((void**)&E_d, nbytes_E));
    h_errchk(hipMalloc((void**)&F_d, nbytes_F));
    
    //// Insert code here to upload arrays D and E //// 
    //// to the compute device                     ////
    
    h_errchk(hipMemcpy(D_d, D_h, nbytes_D, hipMemcpyHostToDevice));
    h_errchk(hipMemcpy(E_d, E_h, nbytes_E, hipMemcpyHostToDevice));

    //// End insert code                           ////

    // Desired block size
    dim3 block_size = { 8, 8, 1 };
    dim3 global_size = { (uint32_t)N0_F, (uint32_t)N1_F, 1 };
    dim3 grid_nblocks;
 
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    // Run the kernel
    hipLaunchKernelGGL(mat_mult, 
            grid_nblocks, 
            block_size, 0, 0, 
            D_d, E_d, F_d,
            N0_F,
            N1_F
    );

    // Wait for any commands to complete on the compute device
    h_errchk(hipDeviceSynchronize());

    //// Step 10. Copy the Buffer for matrix C back to the host ////
    h_errchk(hipMemcpy(F_h, F_d, nbytes_F, hipMemcpyDeviceToHost));
    
    // Write out the result to file
    h_write_binary(F_h, "array_F.dat", nbytes_F);

    // Free the HIP buffers
    h_errchk(hipFree(D_d));
    h_errchk(hipFree(E_d));
    h_errchk(hipFree(F_d));

    // Clean up memory that was allocated on the read   
    free(D_h);
    free(E_h);
    free(F_h);
    
    // Clean up devices
    h_release_devices(num_devices);
}

