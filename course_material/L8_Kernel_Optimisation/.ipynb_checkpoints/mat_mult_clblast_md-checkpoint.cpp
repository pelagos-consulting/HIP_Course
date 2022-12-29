/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <iostream>
#include "omp.h"

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
    
    // Make Buffers on the compute device for matrices A, B, and C
    float_type* array_C = (float_type*)h_alloc(nbytes_C);

    // Allocate memory for buffers
    cl_mem* buffers_A = (cl_mem*)malloc(num_devices*sizeof(cl_mem));
    cl_mem* buffers_B = (cl_mem*)malloc(num_devices*sizeof(cl_mem));    
    cl_mem* buffers_C = (cl_mem*)malloc(num_devices*sizeof(cl_mem));      
    
    for (cl_uint n=0; n<num_devices; n++) {
        
        // Report on the device
        h_report_on_device(devices[n]);
        
        // Create buffers for A
        buffers_A[n] = clCreateBuffer(
            contexts[n], 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            nbytes_A, 
            (void*)array_A, 
            &errcode
        );
        h_errchk(errcode, "Creating buffer_A");

        // Create buffers for B        
        buffers_B[n] = clCreateBuffer(
            contexts[n], 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            nbytes_B, 
            (void*)array_B, 
            &errcode
        );
        h_errchk(errcode, "Creating buffer_B");

        // Create buffers for C
        buffers_C[n] = clCreateBuffer(
            contexts[n], 
            CL_MEM_READ_WRITE, 
            nbytes_C, 
            NULL, 
            &errcode
        );
        h_errchk(errcode, "Creating buffer_C");        
    }

    // Constants for multiplication
    const float alpha=1.0;
    const float beta=0.0;
    
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
    
    // Number of domains along each dimension
    cl_uint D0=4;
    cl_uint D1=4;
    
    // Make maximum local domain sizes
    size_t L0=(size_t)ceil((double)N0_C/(double)D0);
    size_t L1=(size_t)ceil((double)N1_C/(double)D1);    
    
    // Set the number of OpenMP threads
    omp_set_num_threads((int)num_devices);
    //omp_set_num_threads(1);
    
    // Loop over experiments
    for (int n=0; n<nstats; n++) {
        // Time t1
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // Loop over domains using dynamic scheduling
        #pragma omp parallel for shared(command_queues, buffers_A, buffers_B, buffers_C, array_C, D0, D1, N0_C, N1_C, L0, L1, nbytes_C) default(none) schedule(dynamic,1)  
        for (int d=0; d<D0*D1; d++) {
        
            // A is of size (m, k)
            // B is of size (k, n)
            // C is of size (m, n)
        
            // Local domain indices
            size_t l0 = d/D1;
            size_t l1 = d%D1;
            
            size_t start0 = l0*L0;
            size_t start1 = l1*L1;
            
            size_t stop0 = std::min((l0+1)*L0,(size_t)N0_C);
            size_t stop1 = std::min((l1+1)*L1,(size_t)N1_C);
        
            // Get the thread ID
            int tid = omp_get_thread_num();
        
            // size of the local domain
            size_t s0 = stop0-start0;
            size_t s1 = stop1-start1;
        
            // starting positions in the matrices
            size_t offset_A = start0*NCOLS_A;
            size_t offset_B = start1;
            size_t offset_C = start0*NCOLS_C+start1;
        
            // Event for the kernel
            cl_event kernel_event;
        
            // Leading dimension is number of elements that forms the biggest stride
            CLBlastStatusCode status = CLBlastSgemm(
                CLBlastLayoutRowMajor,
                CLBlastTransposeNo,
                CLBlastTransposeNo,
                // Size of region in dim 0 of C
                (const size_t)s0,
                // Size of region in dim 1 of C
                (const size_t)s1,
                // Size of region in dim 1 of A
                (const size_t)NCOLS_A,
                alpha,
                buffers_A[tid], (const size_t)offset_A, (const size_t)NCOLS_A,
                buffers_B[tid], (const size_t)offset_B, (const size_t)NCOLS_C,
                beta,
                buffers_C[tid], (const size_t)offset_C, (const size_t)NCOLS_C,
                &command_queues[tid],
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
            
           // Copy the domain back to C
            
            // B is of size (N1_A, N1_C)
            // Offset is in bytes, row_id and slice_id are indices
            size_t offset=start1*sizeof(cl_float), row_id=start0, slice_id = 0;
    
            // Make up the origin for host and buffer
            const size_t buffer_origin[] = {offset, row_id, slice_id};
            const size_t host_origin[] = {offset, row_id, slice_id};
    
            // Length of a row (in bytes)
            size_t buffer_row_pitch = NCOLS_C * sizeof(cl_float); 
            size_t host_row_pitch = buffer_row_pitch;
    
            // Number of bytes in a slice 
            size_t buffer_slice_pitch = NROWS_C * NCOLS_C * sizeof(cl_float);
            size_t host_slice_pitch = buffer_slice_pitch;        
        
            /// Size of the region to copy, of course we only copy 1 slice
            size_t nrows = s0, nslices = 1;
            const size_t region[] = { s1*sizeof(cl_float), nrows, nslices};
     
            cl_float zero=1.0;
            
            // Enqueue the rectangular copy
            //h_errchk(
            //    clEnqueueReadBufferRect(
            //        command_queues[tid],
            //        buffers_C[tid],
            //        CL_TRUE,
            //        buffer_origin,
            //        host_origin,
            //        region,
            //        buffer_row_pitch,
            //        buffer_slice_pitch,
            //        host_row_pitch,
            //        host_slice_pitch,
            //        array_C,
            //        0,
            //        NULL,
            //        NULL
            //    ),
            //    "Rectangular copy to buffer_B from the host"
            //);
        } // End of parallel region
        
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // Time in milliseconds
        cl_double time_ms = (cl_double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0;
        
        // Keep track of maximum time
        if (time_ms > max_time_ms) {
            max_time_ms = time_ms;
            max_time_n = n;
        }
        
        // Fetch parallel times
        times_ms[n]=time_ms;
        avg_time_ms+=time_ms;
    }
    
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

    for (cl_uint n=0; n<num_devices; n++) {
        // Free the OpenCL buffers
        h_errchk(
            clReleaseMemObject(buffers_A[n]),
            "releasing buffer A"
        );
        h_errchk(
            clReleaseMemObject(buffers_B[n]),
            "releasing buffer B"
        );
        h_errchk(
            clReleaseMemObject(buffers_C[n]),
            "releasing buffer C"
        );
    }
 
    // Free the buffers arrays
    free(buffers_A);
    free(buffers_B);
    free(buffers_C);
    
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

