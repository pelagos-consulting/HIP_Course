    //// Insert missing kernel code ////
    /// To perform Hadamard matrix multiplication ///
    
    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_F) && (i1<N1_F)) {
        
        // Create an offset
        size_t offset = i0*N1_F+i1;
        
        // Number of rows in C is same as number of rows in A
        F[offset]=D[offset]*E[offset];
    }

    //// End insert code ////

