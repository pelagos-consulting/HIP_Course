    // Task 1, build the kernel

    // Reconstruct size of the kernel
    size_t len0_kern = pad0_l + pad0_r + 1;
    size_t len1_kern = pad1_l + pad1_r + 1;

    if ((i0 >= pad0_l) && (i0 < len0_src-pad0_r) 
        && (i1 >= pad1_l) && (i1 < len1_src-pad1_r)) {
        
        // Temporary sum
        float_type sum = 0.0;
        
        // Loop over the kernel
        for (size_t k0 = 0; k0<len0_kern; k0++) {
            for (size_t k1 = 0; k1<len1_kern; k1++) {
                sum+=kern[k0*len1_kern+k1]
                    *src[(i0-pad0_l+k0)*len1_src+(i1-pad1_l+k1)];
            }
        }
        dst[i0*len1_src+i1] = sum;
    }