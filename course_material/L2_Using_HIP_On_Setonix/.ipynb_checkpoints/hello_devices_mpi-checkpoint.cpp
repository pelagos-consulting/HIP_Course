#include <mpi.h>
#include <cassert>

// Include the HIP helper headers
#include "hip_helper.hpp"

// Length of our grid
#define N0_A 512

// Declare the kernel function header
__global__ void fill (float*, float, size_t);

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
    
    // Get the number of HIP devices
    int num_devices=0;
    H_ERRCHK(hipGetDeviceCount(&num_devices));

    // Report on available devices
    for (int n=0; n<num_devices; n++) {
        std::cout << "device " << n << " of " << num_devices << std::endl;
        h_report_on_device(n);
    }
    
    // Set the compute device to use
    H_ERRCHK(hipSetDevice(nranks%num_devices));
    
    // Allocate memory on the compute device for vector A
    float* A_d;
    size_t nbytes_A = N0_A*sizeof(float);
    H_ERRCHK(hipMalloc((void**)&A_d, nbytes_A));
    
    // Allocate memory on the host to hold the filled array
    float* A_h = (float*)calloc(nbytes_A, 1);
    
    // Launch the kernel
    
    // Size of the block in each dimension
    dim3 block_size = { 64, 1, 1};
    // Number of blocks in each dimension
    dim3 grid_nblocks = { N0_A/block_size.x, 1, 1 };
    
    // The value to fill
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
    
    // Release devices and contexts
    h_reset_devices(num_devices);
    
    // Free host memory
    free(A_h);
    
    // End the MPI application
    MPI_Finalize();
}
