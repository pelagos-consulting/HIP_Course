///
/// @file  hello_devices_mpi.cpp
/// 
/// @brief Example program for using MPI with HIP
///
/// Written by Dr. Toby Potter 
/// for the Commonwealth Scientific and Industrial Research Organisation of Australia (CSIRO).
///

#include <cassert>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Include the HIP helper headers
#include <hip/hip_runtime.h>

// Length of our grid
#define N0_A 512

// Declare the kernel function header
__global__ void fill (float*, float, size_t);

// Examine an error code and exit if necessary.
void h_errchk(hipError_t errcode, const char* message) {

    if (errcode != hipSuccess) { 
        const char* errstring = hipGetErrorString(errcode); 
        std::fprintf( 
            stderr, 
            "Error, HIP call failed at %s, error string is: %s\n", 
            message, 
            errstring 
        ); 
        exit(EXIT_FAILURE); 
    }
}

// Macro to check error codes.
#define H_ERRCHK(cmd) \
{\
    h_errchk(cmd, "__FILE__:__LINE__");\
}

// Main program
int main(int argc, char** argv) {
    
    // Initialise MPI
    int ierr = MPI_Init (&argc, &argv);
    assert(ierr==0);
    
    // Get the number of ranks
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
    // Get the MPI rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Print which rank we are using
    std::cout << "MPI rank " << rank << " of " << nranks << "\n";
    
    // Initialise HIP explicitly
    H_ERRCHK(hipInit(0));
    
    // Get the number of HIP devices
    int num_devices=0;
    H_ERRCHK(hipGetDeviceCount(&num_devices));

    // Report on available devices
    for (int i=0; i<num_devices; i++) {
        // Set the compute device to use, 
        // this also makes sure a primary context is available
        H_ERRCHK(hipSetDevice(i));
        
        // Report some information on a compute device
        hipDeviceProp_t prop;

        // Get the properties of the compute device
        H_ERRCHK(hipGetDeviceProperties(&prop, i));

        // ID of the compute device
        std::printf("Device id: %d\n", i);

        // Name of the compute device
        std::printf("\t%-19s %s\n","name:", prop.name);

        // Size of global memory
        std::printf("\t%-19s %lu MB\n","global memory size:",prop.totalGlobalMem/(1000000)); 
    }
    
    // Set the compute device to use
    H_ERRCHK(hipSetDevice(rank%num_devices));
    
    // Allocate memory on the compute device for vector A
    float* A_d;
    size_t nbytes_A = N0_A*sizeof(float);
    H_ERRCHK(hipMalloc((void**)&A_d, nbytes_A));
    
    // Allocate memory on the host for vector A
    float* A_h = (float*)calloc(nbytes_A, 1);
    
    // Launch the kernel
    
    // Size of the block in each dimension
    dim3 block_size = { 64, 1, 1};
    // Number of blocks in each dimension
    dim3 grid_nblocks = { N0_A/block_size.x, 1, 1 };
    
    // The value to fill the vector with
    float fill_value=1.0;
    
    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=0;
    
    // Launch the kernel using hipLaunchKernelGGL method
    hipLaunchKernelGGL(fill, 
            grid_nblocks, 
            block_size, sharedMemBytes, 0, 
            A_d,
            fill_value,
            (size_t)N0_A
    );
    
    // Wait for any commands to complete on the compute device
    H_ERRCHK(hipDeviceSynchronize());
    
    // Download the vector from the compute device
    H_ERRCHK(hipMemcpy((void*)A_h, (const void*)A_d, nbytes_A, hipMemcpyDeviceToHost));
    
    // Free the allocation of memory on the compute device
    H_ERRCHK(hipFree(A_d));
    
    // Check the memory allocation to see if it was filled correctly
    for (int i0=0; i0<N0_A; i0++) {
        assert(A_h[i0]==fill_value);
    }
    
    // Reset devices to clean up resources
    for (int i = 0; i<num_devices; i++) {
        // Set device
        H_ERRCHK(hipSetDevice(i));

        // Synchronize device 
        H_ERRCHK(hipDeviceSynchronize());

        // Reset device (destroys primary context)
        H_ERRCHK(hipDeviceReset());
    }
    
    // Free host memory
    free(A_h);
    
    // End the MPI application
    MPI_Finalize();
}
