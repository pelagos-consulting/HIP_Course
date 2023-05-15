    // Reconstruct size of the kernel
    size_t len0_kern = pad0_l + pad0_r + 1;
    size_t len1_kern = pad1_l + pad1_r + 1;

    // Strides for the source and destination arrays
    size_t stride0_src = len1_src;
    size_t stride1_src = 1;

    // Strides for the cross-correlation kernel
    size_t stride0_kern = len1_kern;
    size_t stride1_kern = 1;

    // Assuming row-major ordering for arrays
    size_t offset_src = i0 * stride0_src + i1;
    size_t offset_kern = pad0_l*stride0_kern + pad1_l*stride1_kern; 

    if ((i0 >= pad0_l) && (i0 < len0_src-pad0_r) && (i1 >= pad1_l) && (i1 < len1_src-pad1_r)) {
        float_type sum = 0.0;
        for (int i = -pad0_l; i<= pad0_r; i++) {
            for (int j = -pad1_l; j <= pad1_r; j++) {
                sum += kern[offset_kern + i*stride0_kern + j*stride1_kern] 
                    * src[offset_src + i*stride0_src + j*stride1_src];
            }
        }
        dst[offset_src] = sum;
    }