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

        
            
cl_int clEnqueueFillBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);


        // Create buffer for the image kernel, copy from host memory image_kernel to fill this
        buffer_kerns[n] = clCreateBuffer(
                contexts[n],
                CL_MEM_COPY_HOST_PTR,
                nbytes_image_kernel,
                (void*)image_kernel,
                &errcode);
        h_errchk(errcode, "Creating buffers for image kernel");

        //// End Task 1 //////////////////////////////////////////////////////////