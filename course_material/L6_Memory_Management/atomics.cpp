/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>
#include <atomic>

// Define the size of the arrays to be computed
#define NROWS_C 256
#define NCOLS_C 256

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

// Kernel to test atomics
__global__ void atomic_test (int *counter) {
    
    // Increment T atomically with device-level synchronisation
    atomicAdd(counter, 1);
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
 
    // Create an integer on the host 
    int counter_h = 0;
    int* counter_d;
    
    //// Step 3. Prepare memory on the device
    
    // Allocate memory for the counter
    H_ERRCHK(hipMalloc((void**)&counter_d, sizeof(int)));
    
    // Zero out the memory
    H_ERRCHK(hipMemset(counter_d, 0, sizeof(int)));
    
    //// Step 4. Launch the kernel
    
    // Launch the kernel
    size_t sharedMemBytes=0;
    
    // Desired block size
    dim3 block_size = { 8, 8, 1 };
    dim3 global_size = { (uint32_t)NCOLS_C, (uint32_t)NROWS_C, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);
    
    hipLaunchKernelGGL(
        atomic_test,
        grid_nblocks,
        block_size,
        sharedMemBytes,
        0,
        counter_d
    );
    
    // Check for errors in the kernel launch
    H_ERRCHK(hipGetLastError());
    
    //// Step 5. Copy the counter back to the host
    
    H_ERRCHK(
        hipMemcpy(
            (void*)&counter_h, 
            (const void*)counter_d, 
            sizeof(int), 
            hipMemcpyDeviceToHost
        )
    );
    
    //// Step 6. Check the result
    
    // Print the counter 
    std::printf(
        "Counter has been incremented %d out of %d times\n", 
        counter_h,
        NROWS_C*NCOLS_C
    );

    //// Step 7. Free resources
    
    H_ERRCHK(hipFree(counter_d));
    
    // Reset compute devices
    h_reset_devices(num_devices);
}

