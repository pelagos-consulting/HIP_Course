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

// Include the CLBLAST library
#include <clblast_c.h>

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
    
    // Number of command queues to generate
    cl_uint num_command_queues = num_devices;
    
    // Do we enable out-of-order execution 
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling?
    cl_bool profiling = CL_TRUE;

    // Do we enable blocking IO?
    cl_bool blocking = CL_TRUE;
    
    // Create the command queues
    cl_command_queue* command_queues = h_create_command_queues(
        devices,
        contexts,
        num_devices,
        num_command_queues,
        ordering,
        profiling
    );

    // Choose the first available context
    // and compute device to use
    assert(dev_index < num_devices);
    cl_context context = contexts[dev_index];
    cl_command_queue command_queue = command_queues[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)
    
    cl_uint N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;
    size_t nbytes_A, nbytes_B, nbytes_C;

    // Read the input data into arrays and sanity check
    float_type* array_A = (float_type*)h_read_binary("array_A.dat", &nbytes_A);
    float_type* array_B = (float_type*)h_read_binary("array_B.dat", &nbytes_B);

    // Sanity check on incoming data
    assert(nbytes_A==N0_C*N1_A*sizeof(float_type));   
    assert(nbytes_B==N1_A*N1_C*sizeof(float_type));
    nbytes_C=N0_C*N1_C*sizeof(float_type);
    
    // Make Buffers on the compute device for matrices A, B, BT, and C
    float_type* array_C = (float_type*)h_alloc(nbytes_C);

    // Make buffer_A by copying from array_A
    cl_mem buffer_A = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        nbytes_A, 
        (void*)array_A, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_A");
    
    // Make buffer_B using array_B as a backing store
    cl_mem buffer_B = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        nbytes_B, 
        (void*)array_B, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_B");
   
    cl_mem buffer_C = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_C, 
                                     NULL, 
                                     &errcode);
    h_errchk(errcode, "Creating buffer_C");

	const float alpha=1.0;
	const float beta=0.0;
	
	cl_event kernel_event;
    
    // Set up a run for clblast
    cl_int nexperiments=1;
    cl_int npoints=2;
    size_t nbytes_output = nexperiments*npoints*sizeof(cl_double);
    cl_double* output_local = (cl_double*)malloc(nbytes_output);    
    
	// Run the experiment nstats times
	size_t nstats=10;
    cl_double times_ms[nstats] = {0};
    cl_double time_ms=0.0;
    cl_double avg_time_ms=0.0;
    cl_double max_time_ms=0.0;
    cl_int max_time_n = 0;
    
    // Run the CLBlast kernel nstats times and collect times
    for (int n=0; n<nstats; n++) {
        
        // Start the clock
        auto t1 = std::chrono::high_resolution_clock::now();
       
        CLBlastStatusCode status = CLBlastSgemm(
            // Choose row-major ordering
            CLBlastLayoutRowMajor,
            // Do we transpose A?
            CLBlastTransposeNo,
            // Do we transpose B?
            CLBlastTransposeNo,
            // Number of rows in C (rows in A) to compute
            (const size_t)NROWS_C,
            // Number of columns in C (columns in B) to compute
            (const size_t)NCOLS_C,
            // Number of columns in A (rows in B) to compute
            (const size_t)NCOLS_A,
            alpha,
            // Buffer, starting offset in elements, length of contiguous dimension
            buffer_A, 0, (const size_t)NCOLS_A,
            buffer_B, 0, (const size_t)NCOLS_C,
            beta,
            buffer_C, 0, (const size_t)NCOLS_C,
            &command_queue,
            &kernel_event
        );
        
        // Make sure the matrix multiplication ran successfully
        assert(status==CLBlastSuccess);
        
        // Wait for events to finish
        h_errchk(
            clWaitForEvents(
                1,
                &kernel_event
            ),
            "Waiting for Sgemm kernels to finish."
        );
        
        // Stop the clock
        auto t2 = std::chrono::high_resolution_clock::now();
        
        cl_double time_ms = (cl_double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0;
        
        if (time_ms > max_time_ms) {
            max_time_ms = time_ms;
            max_time_n = n;
        }
        
        times_ms[n]=time_ms;
        
        avg_time_ms+=time_ms;
    }
    
    // Read memory from the buffer to the host
    h_errchk(
        clEnqueueReadBuffer(command_queue,
                            buffer_C,
                            blocking,
                            0,
                            nbytes_C,
                            array_C,
                            0,
                            NULL,
                            NULL), 
        "Copying matrix C from device to host"
    );
    
    // Calculate the mean and average times
    
    // Leave the longest time out of the calculation
    avg_time_ms = avg_time_ms - max_time_ms;
    avg_time_ms/=(cl_double)(nstats-1);
    cl_double std_time_ms=0.0, scratch=0.0;
    
    for (int n=0; n<nstats; n++) {
        scratch=times_ms[n]-avg_time_ms;
        if (n!=max_time_n) {
            std_time_ms+=(scratch*scratch);
        }
    }
    std_time_ms=sqrt(std_time_ms)/(cl_double)(nstats-1);
    
    output_local[0]=avg_time_ms;
    output_local[1]=std_time_ms;
    
    h_write_binary(output_local, "output_local.dat", nbytes_output);
    free(output_local);

    // Write out the result to file
    h_write_binary(array_C, "array_C.dat", nbytes_C);

    // Free the OpenCL buffers
    h_errchk(
        clReleaseMemObject(buffer_A),
        "releasing buffer A"
    );
    h_errchk(
        clReleaseMemObject(buffer_B),
        "releasing buffer B"
    );
    h_errchk(
        clReleaseMemObject(buffer_C),
        "releasing buffer C"
    );
    
    // Clean up memory that was allocated on the read   
    free(array_A);
    free(array_B);
    free(array_C);
    
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

