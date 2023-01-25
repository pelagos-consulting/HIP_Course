// Kernel to solve the wave equation with fourth-order accuracy in space
__kernel void wave2d_4o (
        __global float* U0,
        __global float* U1,
        __global float* U2,
        __global float* V,
        unsigned int N0,
        unsigned int N1,
        float dt2,
        float inv_dx02,
        float inv_dx12,
        // Position, frequency, and time for the
        // wavelet injection
        unsigned int P0,
        unsigned int P1,
        float pi2fm2t2) {    

    // U2, U1, U0, V is of size (N0, N1)
    size_t i0=get_global_id(1); // Slowest dimension
    size_t i1=get_global_id(0); // Fastest dimension
    
    // Required padding and coefficients for spatial finite difference
    const int pad_l=2, pad_r=2, ncoeffs=5;
    float coeffs[ncoeffs] = {-0.083333336f, 1.3333334f, -2.5f, 1.3333334f, -0.083333336f};
    
    // Limit i0 and i1 to the region of U2 within the padding
    i0=min(i0, (size_t)(N0-1-pad_r));
    i1=min(i1, (size_t)(N1-1-pad_r));
    i0=max((size_t)pad_l, i0);
    i1=max((size_t)pad_l, i1);
    
    // Position within the grid as a 1D offset
    long offset=i0*N1+i1;
    
    // Temporary storage
    float temp0=0.0f, temp1=0.0f;
    float tempV=V[offset];
    
    // Calculate the Laplacian
    #pragma unroll
    for (long n=0; n<ncoeffs; n++) {
        // Stride in dim0 is N1        
        temp0+=coeffs[n]*U1[offset+(n*(long)N1)-(pad_l*(long)N1)];
        // Stride in dim1 is 1
        temp1+=coeffs[n]*U1[offset+n-pad_l];
    }
    
    // Calculate the wavefield U2 at the next timestep
    U2[offset]=(2.0f*U1[offset])-U0[offset]+((dt2*tempV*tempV)*(temp0*inv_dx02+temp1*inv_dx12));
    
    // Inject the forcing term at coordinates (P0, P1)
    if ((i0==P0) && (i1==P1)) {
        U2[offset]+=(1.0f-2.0f*pi2fm2t2)*exp(-pi2fm2t2);
    }
    
}