            //// Begin Task 5 - Code to enqueue the kernel ///////////////////////////
            
            // Enqueue the kernel kernels[tid] using command_queues[tid]
            // local_size, and global_size
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