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