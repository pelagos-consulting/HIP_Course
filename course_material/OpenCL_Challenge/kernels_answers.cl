__kernel void xcorr(
        __global float *src, 
        __global float *dst,
        __global float *kern,
        int len0_src,
        int len1_src, 
        int pad0_l,
        int pad0_r,
        int pad1_l,
        int pad1_r      
    ) {

    // Perform a cross-correlation
    size_t gid0 = get_global_id(0);
    size_t gid1 = get_global_id(1);

    // Reconstruct size of the kernel function
    size_t len0_kern = pad0_l + pad0_r + 1;
    size_t len1_kern = pad1_l + pad1_r + 1;

    // Strides for the source and destination arrays
    long stride0_src = len1_src;
    long stride1_src = 1;

    // Strides for the cross-correlation kernel
    long stride0_kern = len1_kern;
    long stride1_kern = 1;

    // Assuming C ordering for arrays
    long offset_src = gid0 * stride0_src + gid1;
    long offset_kern = pad0_l*stride0_kern + pad1_l*stride1_kern; 

    if ((gid0 >= pad0_l) && (gid0 < len0_src-pad0_r) && (gid1 >= pad1_l) && (gid1 < len1_src-pad1_r)) {
        float sum = 0.0;
        for (int i = -pad0_l; i<= pad0_r; i++) {
            for (int j = -pad1_l; j <= pad1_r; j++) {
                sum += kern[offset_kern + i*stride0_kern + j*stride1_kern] 
                    * src[offset_src + i*stride0_src + j*stride1_src];
            }
        }
        dst[offset_src] = sum;
    }
}
