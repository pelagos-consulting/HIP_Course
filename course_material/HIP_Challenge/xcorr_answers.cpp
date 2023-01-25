// Main

#include <assert.h>
#include "cl_helper.hpp"
#include <cstdio>
#include <omp.h>
#include <chrono>

#include "mat_size.hpp"

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
    cl_bool blocking = CL_TRUE;
    
    // Make a command queue and report on devices
    for (cl_uint n=0; n<num_devices; n++) {
        h_report_on_device(devices[n]);
    }
   
    // Create command queues, one for each device 
    cl_uint num_command_queues = num_devices;
    cl_command_queue* command_queues = h_create_command_queues(
            devices,
            contexts,
            num_devices,
            num_command_queues,
            ordering,
            profiling);

    // Number of Bytes for a single image
    size_t nbytes_image = N0*N1*sizeof(float_type);

    // Number of Bytes for the stack of images
    size_t nbytes_input=NIMAGES*nbytes_image;
    // Output stack is the same size as the input
    size_t nbytes_output=nbytes_input;
    
    // Allocate storage for the output 
    float_type* images_out = (float_type*)h_alloc(nbytes_output);
    
    // Assume that images_in will have dimensions (NIMAGES, N0, N1) and will have row-major ordering
    size_t nbytes;
    
    // Read in the images
    float_type* images_in = (float_type*)h_read_binary("images_in.dat", &nbytes);
    assert(nbytes == nbytes_input);

    // Read in the image kernel
    size_t nbytes_image_kernel = (L0+R0+1)*(L1+R1+1)*sizeof(float_type);
    float_type* image_kernel = (float_type*)h_read_binary("image_kernel.dat", &nbytes);
    assert(nbytes == nbytes_image_kernel);

    // Read kernel sources 
    const char* filename = "kernels_answers.cl";
    char* source = (char*)h_read_binary(filename, &nbytes);

    // Create Programs and kernels for all devices 
    cl_program *programs = (cl_program*)calloc(num_devices, sizeof(cl_program));
    cl_kernel *kernels = (cl_kernel*)calloc(num_devices, sizeof(cl_kernel));
    
    const char* compiler_options = "";
    for (cl_uint n=0; n<num_devices; n++) {
        // Make the program from source
        programs[n] = h_build_program(source, contexts[n], devices[n], compiler_options);
        // And make the kernel
        kernels[n] = clCreateKernel(programs[n], "xcorr", &errcode);
        h_errchk(errcode, "Making a kernel");
    }

    // Create OpenCL buffer for source, destination, and image kernel
    cl_mem *buffer_srces = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
    cl_mem *buffer_dests = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
    cl_mem *buffer_kerns = (cl_mem*)calloc(num_devices, sizeof(cl_mem));
   
    // Create input buffers for every device
    for (cl_uint n=0; n<num_devices; n++) {
        
        //// Begin Task 1 - Code to create the OpenCL buffers for each thread ////
        
        // Fill buffer_srces[n], buffer_dests[n], buffer_kerns[n] 
        // with buffers created by clCreateBuffer 
        // Use the h_errchk routine to check output
        
        // buffer_srces[n] is of size nbytes_image
        // buffer_dests[n] is of size nbytes_image
        // buffer_kerns[n] is of size nbytes_image_kernel
        
        // the array image_kernel contains the host-allocated 
        // memory for the image kernel
        
        // Create buffers for sources
        buffer_srces[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                nbytes_image,
                NULL,
                &errcode);
        h_errchk(errcode, "Creating buffers for sources");

        // Create buffers for destination
        buffer_dests[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_READ_WRITE,
                nbytes_image,
                NULL,
                &errcode);
        h_errchk(errcode, "Creating buffers for destinations");
        
        // Zero out the contents of buffers_dests[n]
        float_type zero=0.0;
        h_errchk(clEnqueueFillBuffer(
                command_queues[n],
                buffer_dests[n],
                &zero,
                sizeof(float_type),
                0,
                nbytes_image,
                0,
                NULL,
                NULL
            ),
            "Filling buffer with zeros."
        );

        // Create buffer for the image kernel, copy from host memory image_kernel to fill this
        buffer_kerns[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_COPY_HOST_PTR,
                nbytes_image_kernel,
                (void*)image_kernel,
                &errcode);
        h_errchk(errcode, "Creating buffers for image kernel");

        //// End Task 1 //////////////////////////////////////////////////////////
        
        //// Begin Task 2 - Code to set kernel arguments for each thread /////////
        
        // Set kernel arguments for kernels[n]
        
        // Just for kernel arguments
        cl_int len0_src = N0, len1_src = N1, pad0_l = L0, pad0_r = R0, pad1_l = L1, pad1_r = R1;
        
        // Set kernel arguments here for convenience
        h_errchk(clSetKernelArg(kernels[n], 0, sizeof(buffer_srces[n]), &buffer_srces[n]), "Set kernel argument 0");
        h_errchk(clSetKernelArg(kernels[n], 1, sizeof(buffer_dests[n]), &buffer_dests[n]), "Set kernel argument 1");
        h_errchk(clSetKernelArg(kernels[n], 2, sizeof(buffer_kerns[n]), &buffer_kerns[n]), "Set kernel argument 2");
        h_errchk(clSetKernelArg(kernels[n], 3, sizeof(cl_int), &len0_src),  "Set kernel argument 3");
        h_errchk(clSetKernelArg(kernels[n], 4, sizeof(cl_int), &len1_src),  "Set kernel argument 4");
        h_errchk(clSetKernelArg(kernels[n], 5, sizeof(cl_int), &pad0_l),    "Set kernel argument 5");
        h_errchk(clSetKernelArg(kernels[n], 6, sizeof(cl_int), &pad0_r),    "Set kernel argument 6");
        h_errchk(clSetKernelArg(kernels[n], 7, sizeof(cl_int), &pad1_l),    "Set kernel argument 7");
        h_errchk(clSetKernelArg(kernels[n], 8, sizeof(cl_int), &pad1_r),    "Set kernel argument 8");
    
        //// End Task 2 //////////////////////////////////////////////////////////
    }
    
    // Start the timer
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Keep track of how many images each device processed
    cl_uint* it_count = (cl_uint*)calloc(num_devices, sizeof(cl_uint)); 
    
    // Make up the local and global sizes to use
    cl_uint work_dim = 2;
    // Desired local size
    const size_t local_size[]={ 16, 16 };
    // Fit the desired global_size
    const size_t global_size[]={ N0, N1 };
    h_fit_global_size(global_size, local_size, work_dim);

    for (cl_uint i = 0; i<NITERS; i++) {
        printf("Processing iteration %d of %d\n", i+1, NITERS);
        
        #pragma omp parallel for default(none) schedule(dynamic, 1) num_threads(num_devices) \
            shared(local_size, global_size, work_dim, images_in, images_out, \
                    buffer_dests, buffer_srces, nbytes_image, \
                    blocking, command_queues, kernels, it_count)
        for (cl_uint n=0; n<NIMAGES; n++) {
            // Get the thread_id
            int tid = omp_get_thread_num();
            
            // Increment image counter for this device
            it_count[tid] += 1;
            
            // Load memory from images in using the offset
            size_t offset = n*N0*N1;
            
            //// Begin Task 3 - Code to upload memory to the compute device buffer ////
            
            // Upload memory from images_in at offset
            // To buffer_srces[tid], using command_queues[tid]
            h_errchk(clEnqueueWriteBuffer(
                        command_queues[tid],
                        buffer_srces[tid],
                        blocking,
                        0,
                        nbytes_image,
                        &images_in[offset],
                        0,
                        NULL,
                        NULL), 
                     "Writing to source buffer"
            );

            //// End Task 3 ///////////////////////////////////////////////////////////
            
            //// Task 4 is to complete the kernel in kernels.cl
            
            //// Begin Task 5 - Code to enqueue the kernel ///////////////////////////
            
            // Enqueue the kernel kernels[tid] using command_queues[tid]
            // work_dim, local_size, and global_size
            h_errchk(clEnqueueNDRangeKernel(
                        command_queues[tid],
                        kernels[tid],
                        work_dim,
                        NULL,
                        global_size,
                        local_size,
                        0, 
                        NULL,
                        NULL), 
                     "Running the xcorr kernel"
            );
            
            //// End Task 5 ///////////////////////////////////////////////////////////
            
            //// Begin Task 6 - Code to download memory from the compute device buffer
            
            //// Download memory buffers_dests[tid] to hosts allocation
            //// images_out at offset
            h_errchk(clEnqueueReadBuffer(
                        command_queues[tid],
                        buffer_dests[tid],
                        blocking,
                        0,
                        nbytes_image,
                        &images_out[offset],
                        0,
                        NULL,
                        NULL), 
                     "Writing to buffer"
            );
            
            //// End Task 6 ///////////////////////////////////////////////////////////
        }
    }

    // Stop the timer
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    double duration = time_span.count();
    
    // Get some statistics on how 
    cl_uint num_images = NITERS*NIMAGES;
    for (cl_uint i = 0; i< num_devices; i++) {
        //h_report_on_device(devices[i]);
        float_type pct = 100*(float_type)it_count[i]/(float_type)num_images;
        printf("Device %d processed %d of %d images (%0.2f%%)\n", i, it_count[i], num_images, pct);
    }
    printf("Overall processing rate %0.2f images/s\n", (double)num_images/duration);

    // Write output data to output file
    h_write_binary(images_out, "images_out.dat", nbytes_output);
    
    // Free allocated memory
    free(source);
    free(image_kernel);
    free(images_in);
    free(images_out);
    free(it_count);

    // Release command queues
    h_release_command_queues(command_queues, num_command_queues);

    // Release programs and kernels
    for (cl_uint n=0; n<num_devices; n++) {
        h_errchk(clReleaseKernel(kernels[n]), "Releasing kernel");
        h_errchk(clReleaseProgram(programs[n]), "Releasing program");
        h_errchk(clReleaseMemObject(buffer_srces[n]),"Releasing sources buffer");
        h_errchk(clReleaseMemObject(buffer_dests[n]),"Releasing dests buffer");
        h_errchk(clReleaseMemObject(buffer_kerns[n]),"Releasing image kernels buffer");
    }

    // Free memory
    free(buffer_srces);
    free(buffer_dests);
    free(buffer_kerns);
    free(programs);
    free(kernels);

    // Release devices and contexts
    h_release_devices(devices, num_devices, contexts, platforms);
}
