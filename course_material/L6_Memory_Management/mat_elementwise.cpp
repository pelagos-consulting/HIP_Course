/* Code to perform a Matrix multiplication using HIP
Written by Dr Toby M. Potter
*/

// Setup headers
#include <cassert>
#include <cmath>
#include <iostream>

// Define the size of the arrays to be computed
#define NROWS_F 8
#define NCOLS_F 4

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

// Bring in a library to manage matrices on the CPU
#include "mat_helper.hpp"

// standard matrix multiply kernel 
__global__ void mat_hadamard (
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
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_F) && (i1<N1_F)) {

        // Offset into the arrays
        size_t offset = i0*N1_F + i1;
        
        // Number of rows in C is same as number of rows in A
        F[offset]=D[offset]*E[offset];
    }
} 

int main(int argc, char** argv) {
    
    //// Step 1. Parse command line arguments ////

    // Parse arguments
    int dev_index = h_parse_args(argc, argv);
    
    // Number of devices discovered
    int num_devices=0;
    
    //// Step 2. Device discovery and selection ////
    
    // Helper function to acquire devices
    // This sets the default device
    h_acquire_devices(&num_devices, dev_index);
        
    // Report on the device in use
    h_report_on_device(dev_index);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // D, E, and F are of size (N0_F, N1_F)
    size_t N0_F = NROWS_F, N1_F = NCOLS_F;

    //// Step 3. Construct matrices D_h, E_h, and F_h on the host 
    //// fill D_h and E_h with random numbers ////
    
    // Number of bytes in each array
    size_t nbytes_D = N0_F*N1_F*sizeof(float);
    size_t nbytes_E = N0_F*N1_F*sizeof(float);
    size_t nbytes_F = N0_F*N1_F*sizeof(float);

    // Allocate pinned memory for the host arrays
    float *D_h, *E_h, *F_h;
    H_ERRCHK(hipHostMalloc((void**)&D_h, nbytes_D));
    H_ERRCHK(hipHostMalloc((void**)&E_h, nbytes_E));
    H_ERRCHK(hipHostMalloc((void**)&F_h, nbytes_F));

    // Fill the host arrays with random numbers 
    // using the matrix helper library
    m_random(D_h, N0_F, N1_F);
    m_random(E_h, N0_F, N1_F);
    
    //// Step 4. Allocate memory for arrays //// 
    //// D_d, E_d, and F_d on the compute device ////

    float *D_d, *E_d, *F_d;
    H_ERRCHK(hipMalloc((void**)&D_d, nbytes_D));
    H_ERRCHK(hipMalloc((void**)&E_d, nbytes_E));
    H_ERRCHK(hipMalloc((void**)&F_d, nbytes_F));

    //// Step 5. 1. Upload matrices D_h and E_h from the host //// 
    //// to D_d and E_d on the device ////
    
    H_ERRCHK(hipMemcpy(D_d, D_h, nbytes_D, hipMemcpyHostToDevice));
    H_ERRCHK(hipMemcpy(E_d, E_h, nbytes_E, hipMemcpyHostToDevice));
 
    //// Step 6. Run the kernel to compute F_d ///
    //// from D_d and E_d on the device ////
        
    // Desired block size
    dim3 block_size = { 3, 3, 1 };
    dim3 global_size = { (uint32_t)N1_F, (uint32_t)N0_F, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=0;
    
    // Launch the kernel using hipLaunchKernelGGL method
    hipLaunchKernelGGL(mat_hadamard, 
            grid_nblocks, 
            block_size, sharedMemBytes, 0, 
            D_d, E_d, F_d,
            N0_F,
            N1_F
    );
    
    // Alternatively, launch the kernel using CUDA triple Chevron syntax
    //mat_hadamard<<<grid_nblocks, block_size, sharedMemBytes, 0>>>(D_d, E_d, F_d, N0_F, N1_F);
    
    // Wait for any commands to complete on the compute device
    H_ERRCHK(hipDeviceSynchronize());

    //// Step 7. Copy the buffer for matrix F_d //// 
    //// on the device back to F_h on the host ////
    
    //// Exercise: Replace the call to hipMemcpy with a 3D copy
    //// through a call to hipMemcpy3D
    
    H_ERRCHK(hipMemcpy((void*)F_h, (const void*)F_d, nbytes_F, hipMemcpyDeviceToHost));
    
    //// Step 8. Test the computed matrix F_h against a known answer
    
    // Check the answer against a known solution
    float* F_answer_h = (float*)calloc(nbytes_F, 1);
    float* F_residual_h = (float*)calloc(nbytes_F, 1);

    // Compute the known solution
    m_hadamard(D_h, E_h, F_answer_h, N0_F, N1_F);

    // Compute the residual between F_h and F_answer_h
    m_residual(F_answer_h, F_h, F_residual_h, N0_F, N1_F);

    // Pretty print the output matrices
    std::cout << "The output array F_h (as computed with HIP) is\n";
    m_show_matrix(F_h, N0_F, N1_F);

    std::cout << "The CPU solution (F_answer_h) is \n";
    m_show_matrix(F_answer_h, N0_F, N1_F);
    
    std::cout << "The residual (F_answer_h-F_h) is\n";
    m_show_matrix(F_residual_h, N0_F, N1_F);

    // Print the maximum error between matrices
    float max_err = m_max_error(F_h, F_answer_h, N0_F, N1_F);
    
    //// Step 9. Write the contents of matrices 
    //// D_h, E_h, and F_h to disk ////

    // Write out the result to file
    h_write_binary(D_h, "array_D.dat", nbytes_D);
    h_write_binary(E_h, "array_E.dat", nbytes_E);
    h_write_binary(F_h, "array_F.dat", nbytes_F);
    
    //// Step 10. Release resources
    
    // Free the HIP buffers
    H_ERRCHK(hipFree(D_d));
    H_ERRCHK(hipFree(E_d));
    H_ERRCHK(hipFree(F_d));

    // Clean up pinned memory on the host   
    H_ERRCHK(hipHostFree(D_h));
    H_ERRCHK(hipHostFree(E_h));
    H_ERRCHK(hipHostFree(F_h));

    // Free the answer and residual matrices
    free(F_answer_h);
    free(F_residual_h);
    
    // Reset compute devices
    h_reset_devices(num_devices);
}

