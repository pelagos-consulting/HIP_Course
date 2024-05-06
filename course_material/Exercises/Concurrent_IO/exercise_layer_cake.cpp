/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <iostream>

// Include the size of arrays to be computed
#include "mat_size.hpp"

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

// Bring in helper library to manage matrices
#include "mat_helper.hpp"

// Kernel to solve the wave equation with fourth-order accuracy in space
__global__ void fill_plane (
        // Arguments
        float_type* U,
        int n,
        size_t N0,
        size_t N1) {    

    // U2, U1, U0, V is of size (N0, N1)
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure i0 and i1 are sane
    i0=min(i0, (N0-1));
    i1=min(i1, (N1-1));
       
    // Fill the solution with the fill value
    U[i0*N1+i1] = (float_type)n;
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
    
    // Number of scratch buffers, must be at least 2
    const size_t nscratch=1;
    
    // Number of streams to create
    const size_t nstream = 1;
    
    // Create streams that block with null stream
    hipStream_t* streams = h_create_streams(nstream, 1);
    hipStream_t active_stream = streams[0];

    // Make up sizes 
    size_t N0=N0_U, N1=N1_U;
    
    // Size of the grid
    size_t nbytes_U=N0*N1*sizeof(float_type);

    // Setup 5 iterations
    size_t NT = 10;
    
    // Make up the output array
    size_t nbytes_out = NT*N0*N1*sizeof(float_type);
    
    // Allocate pinned memory for out_h
    // So we can enable asynchronous copies
    float_type* out_h;
    H_ERRCHK(
        hipHostMalloc(
            (void**)&out_h, 
            nbytes_out, 
            hipHostMallocDefault
        )
    );
    
    // Reset memory
    H_ERRCHK(hipMemset(out_h, 0, nbytes_out));
    
    // Make buffers on the compute device for the solution U
    
    // Create events for maintaining sync
    hipEvent_t events[nscratch];
    
    // Create scratch buffers for the computation
    float_type* U_ds[nscratch] = {NULL};
    for (int n=0; n<nscratch; n++) {
        // Allocate memory and zero out
        H_ERRCHK(hipMalloc((void**)&U_ds[n], nbytes_U));
        H_ERRCHK(hipMemset(U_ds[n], 0, nbytes_U));
        
        // Initialise the event
        H_ERRCHK(hipEventCreate(&events[n]));
    }
    
    // Desired block size
    dim3 block_size = { 4, 4, 1 };
    dim3 global_size = { (uint32_t)N1, (uint32_t)N0, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=0;
    
    // Setup parameters for an asynchronous 3D memory copy
    
    // For the host
    hipPitchedPtr out_h_ptr = make_hipPitchedPtr(
        out_h, // pointer 
        N1*sizeof(float), // pitch - actual pencil width (bytes) 
        N1, // pencil width (elements)
        N0 // number of pencils in a plane (elements), not the total number of pencils
    );
    // For the device
    hipPitchedPtr out_d_ptr = make_hipPitchedPtr(
        U_ds[0], // pointer
        N1*sizeof(float), // pitch - actual pencil width (bytes) 
        N1, // pencil width (elements)
        N0 // number of pencils in a plane (elements), not the total number of pencils
    );
    // Postion within the host array
    hipPos out_h_pos = make_hipPos(
        0*sizeof(float), // byte position along a pencil (bytes)
        0, // starting pencil index (elements)
        0 // start pencil plane index (elements)
    );
    // Postion within the device array
    hipPos out_d_pos = make_hipPos(
        0*sizeof(float), // byte position along a pencil (bytes)
        0, // starting pencil index (elements)
        0 // starting pencil plane index (elements)
    );
    // Choose the region to copy
    hipExtent extent = make_hipExtent(
        N1*sizeof(float), // width of pencil region to copy (bytes)
        N0, // number of pencils to copy the region from
        1 // number of pencil planes
    );

    // Fill the copy parameters
    hipMemcpy3DParms copy_parms = {0};
    copy_parms.srcPtr = out_d_ptr;
    copy_parms.srcPos = out_d_pos;
    copy_parms.dstPtr = out_h_ptr;
    copy_parms.dstPos = out_h_pos;
    copy_parms.extent = extent;
    copy_parms.kind = hipMemcpyDeviceToHost;
    
    // Start the clock
    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (size_t n=0; n<NT; n++) {
        
        // Launch the kernel using hipLaunchKernelGGL method
        // Use 0 when choosing the default (null) stream
        hipLaunchKernelGGL(fill_plane, 
            grid_nblocks, block_size, sharedMemBytes, 0,
            U_ds[n%nstream],
            n, N0, N1
        );
                           
        // Check the status of the kernel launch
        H_ERRCHK(hipGetLastError());
           
        // Read memory from the buffer to the host in an asynchronous manner
           
        // Only change what is necessary in copy_parms
        copy_parms.srcPtr.ptr = U_ds[n%nscratch];
            
        // Z positions equal to 1 don't seem to work on AMD platforms?!?!
        copy_parms.dstPos.z = n;
        
        if (n!=1) {
            //H_ERRCHK(
            //    hipMemcpy(
            //        &out_h[n*N0*N1],
            //        U_ds[n%nscratch],
            //        nbytes_U,
            //        hipMemcpyDeviceToHost
            //    )
            //);
            
            // Copy memory synchronously
            H_ERRCHK(
                hipMemcpy3D(
                    &copy_parms
                )
            );
        }
    }

    // Make sure all work is done
    H_ERRCHK(hipDeviceSynchronize());
    
    // Stop the clock
    auto t2 = std::chrono::high_resolution_clock::now();    
    double time_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0;
    printf("The synchronous calculation took %f milliseconds.\n", time_ms);
    
    // Write out the result to file
    h_write_binary(out_h, "layer_cake_out.dat", nbytes_out);

    // Test by printing each layer 
    // of the cake for consistency    
    for (size_t n=0; n<NT; n++) {
        size_t offset=n*N0*N1;
    
        // Print the layer in the output array
        m_show_matrix(&out_h[offset], N0, N1);
    }
    
    // Free resources
    for (int n=0; n<nscratch; n++) {
        H_ERRCHK(hipFree(U_ds[n]));
    }
    
    // Free out_h on the host
    H_ERRCHK(hipHostFree(out_h));
    
    // Release compute streams
    h_release_streams(nstream, streams);
    
    // Reset compute devices
    h_reset_devices(num_devices);

    return 0;
}

