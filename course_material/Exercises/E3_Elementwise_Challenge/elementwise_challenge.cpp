/* Code to perform Hadamard (elementwise) multiplication using HIP
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

// Size of the problem
#include "mat_size.hpp"

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

// Bring in helper header to work with matrices
#include "mat_helper.hpp"

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

    //// Step 1. Insert missing kernel code ////
    //// To perform Hadamard matrix multiplication ////
  
    // Uncomment for the shortcut answer
    // #include "step1_kernel.hpp" 

    //// End code: ////
} 

int main(int argc, char** argv) {
    
    // Parse arguments and choose the index of the target device
    int dev_index = h_parse_args(argc, argv);
    
    // Number of devices discovered
    int num_devices=0;

    //// Step 2. Discover resources //// 
    //// Call hipInit to intialise HIP ////
    //// Call hipGetDeviceCount to fill num_devices ////
    //// Make sure dev_index is sane
    //// Call hipSetDevice to set the compute device ///

    // Uncomment for the shortcut answer
    // #include "step2_resources.hpp"
        
    //// End code: ////
        
    // Report on the device in use
    h_report_on_device(dev_index);

    // Matrices D, E, and F are of size (N0_F, N1_F)
    size_t N0_F = NROWS_F, N1_F = NCOLS_F;

    // Number of bytes in each matrix
    size_t nbytes_D=N0_F*N1_F*sizeof(float);   
    size_t nbytes_E=N0_F*N1_F*sizeof(float);
    size_t nbytes_F=N0_F*N1_F*sizeof(float);
    
    // Allocate host matrices using pinned memory
    float *D_h, *E_h, *F_h; 
    H_ERRCHK(hipHostMalloc((void**)&D_h, nbytes_D));
    H_ERRCHK(hipHostMalloc((void**)&E_h, nbytes_E));
    H_ERRCHK(hipHostMalloc((void**)&F_h, nbytes_F));
 
    // Fill host matrices with random numbers in the range 0, 1
    m_random(D_h, N0_F, N1_F);
    m_random(E_h, N0_F, N1_F);

    //// Step 3. Use hipMalloc to allocate memory
    //// for arrays D_d, E_d, and F_d ////
    //// on the compute device ////

    // Uncomment for the shortcut answer
    // #include "step3_device_allocation.hpp"

    //// End code: ////

    //// Step 4. Upload matrices ////
    //// Use hipMemcpy to upload arrays ////
    //// D_h and E_h on the host ////
    //// to D_d and E_d on the compute device //// 

    // Uncomment for the shortcut answer
    // #include "step4_memcpy_todevice.hpp"

    //// End code:  ////

    // Desired block size
    dim3 block_size = { 2, 2, 1 };
    dim3 global_size = { (uint32_t)N1_F, (uint32_t)N0_F, 1 };
    dim3 grid_nblocks;
 
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    //// Step 5. Launch the kernel ////
    //// Use hipLaunchKernelGGL to launch the kernel ////
    //// Use hipDeviceSynchronize to wait on the kernel

    // Uncomment for the shortcut answer
    // #include "step5_kernel_launch.hpp"

    //// End code:  ////

    //// Step 6. Copy the solution back from the compute device ////
    //// Use hipMemcpy to copy F_d on the device ////
    //// back to F_h on the host ////

    // Uncomment for the shortcut answer
    // #include "step6_memcpy_fromdevice.hpp"

    //// End code: ////

    // Check the answer against a known solution
    float* F_answer_h = (float*)calloc(nbytes_F, 1);
    float* F_residual_h = (float*)calloc(nbytes_F, 1);

    // Compute the known solution
    m_hadamard(D_h, E_h, F_answer_h, N0_F, N1_F);

    // Compute the residual between F_h and F_answer_h
    m_residual(F_answer_h, F_h, F_residual_h, N0_F, N1_F);

    // Pretty print matrices
    std::cout << "The output array F_h (as computed with HIP) is\n";
    m_show_matrix(F_h, N0_F, N1_F);

    std::cout << "The CPU solution (F_answer_h) is \n";
    m_show_matrix(F_answer_h, N0_F, N1_F);
    
    std::cout << "The residual (F_answer_h-F_h) is\n";
    m_show_matrix(F_residual_h, N0_F, N1_F);

    // Write out the result to file
    h_write_binary(D_h, "array_D.dat", nbytes_D);
    h_write_binary(E_h, "array_E.dat", nbytes_E);
    h_write_binary(F_h, "array_F.dat", nbytes_F);

    //// Step 7. Free memory on device allocations D_d, E_d and F_d ////
    //// Use hipFree to free device memory ////
    
    // Uncomment for the shortcut answer
    // #include "step7_free_devicemem.hpp"

    //// End code: ////

    // Free the pinned host memory
    H_ERRCHK(hipHostFree(D_h));
    H_ERRCHK(hipHostFree(E_h));
    H_ERRCHK(hipHostFree(F_h));

    // Clean up memory that was allocated on the host using calloc
    free(F_answer_h);
    free(F_residual_h);

    //// Step 8. Free resources ////
    //// Use hipDeviceSychronize to finish all work on the compute device ////
    //// Use hipDeviceReset to reset the compute device

    //// Uncomment for the shortcut answer
    // #include "step8_release_resources.hpp"

    //// End code: ////
}

