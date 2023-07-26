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

typedef float float_type;

// Kernel to solve the wave equation using 
// finite differencing that is accurate 
// to second-order in time, 
// and fourth order in space
__global__ void wave2d_4o (
        // Arguments
        float_type* U0,
        float_type* U1,
        float_type* U2,
        float_type* V,
        size_t N0,
        size_t N1,
        float dt2,
        float inv_dx02,
        float inv_dx12,
        // Position, frequency, and time for the
        // wavelet injection
        size_t P0,
        size_t P1,
        float pi2fm2t2) {    

    // U2, U1, U0, V is of size (N0, N1)
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Required padding and coefficients for spatial finite difference
    const int pad_l=2, pad_r=2, ncoeffs=5;
    float coeffs[ncoeffs] = {-0.083333336f, 1.3333334f, -2.5f, 1.3333334f, -0.083333336f};
    
    // Limit i0 and i1 to the region of U2 within the padding
    i0=min(i0, (size_t)(N0-1-pad_r));
    i1=min(i1, (size_t)(N1-1-pad_r));
    i0=max((size_t)pad_l, i0);
    i1=max((size_t)pad_l, i1);
    
    // Position within the grid as a 1D offset
    long offset=i0*N1+i1;
    
    // Temporary storage
    float temp0=0.0f, temp1=0.0f;
    float tempV=V[offset];
    
    // Calculate the Laplacian
    for (long n=0; n<ncoeffs; n++) {
        // Stride in dim0 is N1        
        temp0+=coeffs[n]*U1[offset+(n*(long)N1)-(pad_l*(long)N1)];
        // Stride in dim1 is 1
        temp1+=coeffs[n]*U1[offset+n-pad_l];
    }
    
    // Calculate the wavefield U2 at the next timestep
    U2[offset]=(2.0f*U1[offset])-U0[offset]+((dt2*tempV*tempV)*(temp0*inv_dx02+temp1*inv_dx12));
    
    // Inject the forcing term at coordinates (P0, P1)
    if ((i0==P0) && (i1==P1)) {
        U2[offset]+=(1.0f-2.0f*pi2fm2t2)*exp(-pi2fm2t2);
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
    
    // Number of scratch buffers, must be at least 3
    const int nscratch=3;
    
    // Make up sizes 
    size_t N0=N0_U, N1=N1_U;
    
    // Size of the grid
    size_t nbytes_U=N0*N1*sizeof(float_type);
    
    // Read in the velocity from disk and find the maximum
    float_type* V_h = (float_type*)h_alloc(nbytes_U);
    float_type Vmax = VEL;
    for (size_t i=0; i<N0*N1; i++) {
        V_h[i] = Vmax;
    }

    // Make up the timestep using maximum velocity
    float_type dt = CFL*std::min(D0, D1)/Vmax;
    printf("dt=%f, Vmax=%f\n", dt, Vmax);
    
    // Use a grid crossing time at maximum velocity to get the number of timesteps
    int NT = (int)std::max(D0*N0, D1*N1)/(dt*Vmax);
    
    // Make up the output array
    size_t nbytes_out = NT*N0*N1*sizeof(float_type);
    float_type* out_h = (float_type*)h_alloc(nbytes_out);
    
    // Make buffers on the compute device for matrices U0, U1, U2, V
    
    // Read-only buffer for V
    float_type* V_d;
    H_ERRCHK(hipMalloc((void**)&V_d, nbytes_U));
    H_ERRCHK(hipMemcpy(V_d, V_h, nbytes_U, hipMemcpyHostToDevice));
    
    // Create scratch buffers for the computation
    float_type* U_ds[nscratch] = {NULL};
    for (int n=0; n<nscratch; n++) {
        // Allocate memory and zero out
        H_ERRCHK(hipMalloc((void**)&U_ds[n], nbytes_U));
        H_ERRCHK(hipMemset(U_ds[n], 0, nbytes_U));
    }
    
    // Set up arguments for the kernel
    float_type dt2=dt*dt, inv_dx02=1.0/(D0*D0), inv_dx12=1.0/(D1*D1);
    
    // time, pi^2 fm^2 t^2 for the Ricker wavelet
    // Get the frequency of the wavelet
    
    // Number of points per wavelength
    float_type ppw=10.0f;
    // Frequency of the Ricker Wavelet
    float_type fm=Vmax/(ppw*std::max(D0,D1));
    float_type pi=3.141592f;
    float_type t=0.0f, pi2fm2t2=0.0f;
    // Min-to-min time of the wavelet
    float_type td=std::sqrt(6.0f)/(pi*fm);
    
    printf("dt=%g, fm=%g, Vmax=%g, dt2=%g\n", dt, fm, Vmax, dt2);
    
    // Coordinates of the Ricker wavelet
    size_t P0=N0/2;
    size_t P1=N1/2;
    
    // Desired block size
    dim3 block_size = { 64, 4, 1 };
    dim3 global_size = { (uint32_t)N1, (uint32_t)N0, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=0;
    
    // Main loop
    float_type *U0_d, *U1_d, *U2_d;
    
    // Start the clock
    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (int n=0; n<NT; n++) {
        // Get the wavefields
        U0_d = U_ds[n%nscratch];
        U1_d = U_ds[(n+1)%nscratch];
        U2_d = U_ds[(n+2)%nscratch];
        
        // Shifted time
        t = n*dt-2.0*td;
        pi2fm2t2 = pi*pi*fm*fm*t*t;
        
        // Launch the kernel using hipLaunchKernelGGL method
        // Use 0 when choosing the default (null) stream
        hipLaunchKernelGGL(wave2d_4o, 
            grid_nblocks, block_size, sharedMemBytes, 0,
            U0_d, U1_d, U2_d, V_d,
            N0, N1, dt2,
            inv_dx02, inv_dx12,
            P0, P1, pi2fm2t2
        );
                           
        // Check the status of the kernel launch
        H_ERRCHK(hipGetLastError());
          
        // Copy the wavefield back to the host
        // hipMemcpy is a barrier
        if (n>1 && n<NT-1) { // For consistency with the async solution
            H_ERRCHK(
                hipMemcpy(
                    (void*)&out_h[n*N0*N1],
                    U0_d,
                    nbytes_U,
                    hipMemcpyDeviceToHost
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
    h_write_binary(out_h, "array_out.dat", nbytes_out);

    // Free the HIP buffers
    H_ERRCHK(hipFree(V_d));

    for (int n=0; n<nscratch; n++) {
        H_ERRCHK(hipFree(U_ds[n]));
    }
    
    // Clean up memory that was allocated on the read   
    free(V_h);
    free(out_h);
    
    // Reset compute devices
    h_reset_devices(num_devices);

    return 0;
}

