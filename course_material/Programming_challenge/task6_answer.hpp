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