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

    // Implement the rest of this kernel

}
