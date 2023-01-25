/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <iostream>

// Include the size of arrays to be computed
#include "mat_size.hpp"

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

typedef cl_float float_type;

int main(int argc, char** argv) {
   
    // Parse arguments and set the target device
    cl_device_type target_device;
    cl_uint dev_index = h_parse_args(argc, argv, &target_device);
    
    // Useful for checking OpenCL errors
    cl_int errcode;

    // Create handles to platforms, 
    // devices, and contexts

    // Number of platforms discovered
    cl_uint num_platforms;

    // Number of devices discovered
    cl_uint num_devices;

    // Pointer to an array of platforms
    cl_platform_id *platforms = NULL;

    // Pointer to an array of devices
    cl_device_id *devices = NULL;

    // Pointer to an array of contexts
    cl_context *contexts = NULL;
    
    // Helper function to acquire devices
    h_acquire_devices(target_device,
                     &platforms,
                     &num_platforms,
                     &devices,
                     &num_devices,
                     &contexts);
    

    
    // Do we enable out-of-order execution 
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling?
    cl_bool profiling = CL_TRUE;

    // Do we enable blocking IO?
    cl_bool blocking = CL_FALSE;
    
    // Number of scratch buffers, must be at least 3
    const int nscratch=5;
    
    // Number of command queues to generate
    cl_uint num_command_queues = nscratch;
    
    // Choose the first available context
    // and compute device to use
    assert(dev_index < num_devices);
    cl_context context = contexts[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Create the command queues
    cl_command_queue* command_queues = h_create_command_queues(
        &device,
        &context,
        (cl_uint)1,
        (cl_uint)(num_command_queues+1),
        ordering,
        profiling
    );
    
    // Command queue to do temporary things
    cl_command_queue compute_queue = command_queues[nscratch];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    size_t nbytes_U;
    
    // Read in the velocity from disk and find the maximum
    float_type* array_V = (float_type*)h_read_binary("array_V.dat", &nbytes_U);
    assert(nbytes_U==N0*N1*sizeof(float_type));
    float_type Vmax = 0.0;
    for (size_t i=0; i<N0*N1; i++) {
        Vmax = (array_V[i]>Vmax) ? array_V[i] : Vmax;
    }

    // Make up the timestep using maximum velocity
    float_type dt = CFL*std::min(D0, D1)/Vmax;
    
    printf("dt=%f, Vmax=%f\n", dt, Vmax);
    
    // Use a grid crossing time at maximum velocity to get the number of timesteps
    int NT = (int)std::max(D0*N0, D1*N1)/(dt*Vmax);
    
    // Make up the output array
    size_t nbytes_out = NT*N0*N1*sizeof(cl_float);
    cl_float* array_out = (cl_float*)h_alloc(nbytes_out);
    
    // Make Buffers on the compute device for matrices U0, U1, U2, V
    
    // Read-only buffer for V
    cl_mem buffer_V = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
        nbytes_U, 
        (void*)array_V, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_V");
    
    // Make up events
    cl_event events[nscratch];
    
    // Create scratch buffers for the computation
    cl_mem buffers_U[nscratch];
    for (int n=0; n<nscratch; n++) {
        buffers_U[n] = clCreateBuffer(
            context, 
            CL_MEM_ALLOC_HOST_PTR, 
            nbytes_U, 
            NULL, 
            &errcode
        );
        h_errchk(errcode, "Creating scratch buffer.");
        
        // Zero out buffers
        float_type zero=0.0f;
        h_errchk(
            clEnqueueFillBuffer(
                compute_queue,
                buffers_U[n],
                &zero,
                sizeof(float_type),
                0,
                nbytes_U,
                0,
                NULL,
                &events[n]
            ),
            "Filling buffer with zeroes."
        );
    }
    
    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "wave2d_4o", &errcode);
    h_errchk(errcode, "Creating wave2d_4o Kernel");

    // Set up arguments for the kernel
    cl_uint N0_k=N0, N1_k=N1;
    cl_float dt2=dt*dt, inv_dx02=1.0/(D0*D0), inv_dx12=1.0/(D1*D1);
    
    // time, pi^2 fm^2 t^2 for the Ricker wavelet
    // Get the frequency of the wavelet
    
    // Number of points per wavelength
    float_type ppw=10;
    // Frequency of the Ricker Wavelet
    float_type fm=Vmax/(ppw*std::max(D0,D1));
    float_type pi=3.141592f;
    float_type t=0.0f, pi2fm2t2=0.0f;
    // Min-to-min time of the wavelet
    float_type td=std::sqrt(6.0f)/(pi*fm);
    
    printf("dt=%g, fm=%g, Vmax=%g, dt2=%g\n", dt, fm, Vmax, dt2);
    
    // Coordinates of the Ricker wavelet
    cl_uint P0=N0/2;
    cl_uint P1=N1/2;
    
    // Set arguments to the kernel (not thread safe)
    h_errchk(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &buffer_V ),
        "setting kernel argument 3"
    );
    h_errchk(
        clSetKernelArg(kernel, 4, sizeof(cl_uint), &N0_k ),
        "setting kernel argument 4"
    );
    h_errchk(
        clSetKernelArg(kernel, 5, sizeof(cl_uint), &N1_k ),
        "setting kernel argument 5"
    );
    h_errchk(
        clSetKernelArg(kernel, 6, sizeof(cl_float), &dt2 ),
        "setting kernel argument 6"
    );
    h_errchk(
        clSetKernelArg(kernel, 7, sizeof(cl_float), &inv_dx02 ),
        "setting kernel argument 7"
    );
    h_errchk(
        clSetKernelArg(kernel, 8, sizeof(cl_float), &inv_dx12 ),
        "setting kernel argument 8"
    );
    h_errchk(
        clSetKernelArg(kernel, 9, sizeof(cl_uint), &P0 ),
        "setting kernel argument 9"
    );
    h_errchk(
        clSetKernelArg(kernel, 10, sizeof(cl_uint), &P1 ),
        "setting kernel argument 10"
    );
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 64, 4 };
    
    // Desired global_size
    const size_t global_size[]={ N1, N0 };
    h_fit_global_size(global_size, local_size, work_dim);
    
    // Main loop
    cl_mem U0, U1, U2;
    
    // Start the clock
    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (int n=0; n<NT; n++) {
        
        // Wait for the previous copy command to finish
        h_errchk(
            clFinish(command_queues[(n+2)%nscratch]),
            "Waiting for all previous things to finish"
        );
        
        // Get the wavefields
        U0 = buffers_U[n%nscratch];
        U1 = buffers_U[(n+1)%nscratch];
        U2 = buffers_U[(n+2)%nscratch];
        
        // Shifted time
        t = n*dt-2.0*td;
        pi2fm2t2 = pi*pi*fm*fm*t*t;
        
        // Set kernel arguments
        h_errchk(
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &U0 ),
            "setting kernel argument 0"
        );
        h_errchk(
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &U1 ),
            "setting kernel argument 1"
        );
        h_errchk(
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &U2 ),
            "setting kernel argument 2"
        );
        h_errchk(
            clSetKernelArg(kernel, 11, sizeof(cl_float), &pi2fm2t2 ),
            "setting kernel argument 11"
        );
        
        // Enqueue the wave solver    
        h_errchk(
            clEnqueueNDRangeKernel(
                compute_queue,
                kernel,
                work_dim,
                NULL,
                global_size,
                local_size,
                0,
                NULL,
                &events[n%nscratch]), 
            "Running the kernel"
        );
          
        // Read memory from the buffer to the host in an asynchronous manner
        if (n>0) {
            cl_int copy_index=n-1;
            h_errchk(
                clEnqueueReadBuffer(
                    command_queues[copy_index%nscratch],
                    buffers_U[copy_index%nscratch],
                    blocking,
                    0,
                    nbytes_U,
                    &array_out[copy_index*N0*N1],
                    1,
                    &events[copy_index%nscratch],
                    NULL), 
                "Asynchronous copy from U2 on device to host"
            );
        }
    }

    // Make sure all work is done
    for (int i=0; i<nscratch; i++) {
        h_errchk(
            clFinish(command_queues[i]),
            "Finishing the command queues."
        );
    }

    // Stop the clock
    auto t2 = std::chrono::high_resolution_clock::now();    
    cl_double time_ms = (cl_double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0;
    printf("The asynchronous calculation took %d milliseconds.", time_ms);
    
    // Write out the result to file
    h_write_binary(array_out, "array_out.dat", nbytes_out);

    // Free the OpenCL buffers
    h_errchk(
        clReleaseMemObject(buffer_V),
        "releasing buffer V"
    );
    for (int n=0; n<nscratch; n++) {
        h_errchk(
            clReleaseMemObject(buffers_U[n]),
            "Releasing scratch buffer"
        );
    }
    
    // Clean up memory that was allocated on the read   
    free(array_V);
    free(array_out);
    
    // Clean up command queues
    h_release_command_queues(
        command_queues, 
        num_command_queues
    );
    
    // Clean up devices, queues, and contexts
    h_release_devices(
        devices,
        num_devices,
        contexts,
        platforms
    );

    return 0;
}

