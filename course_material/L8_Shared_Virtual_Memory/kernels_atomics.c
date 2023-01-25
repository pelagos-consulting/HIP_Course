// Kernel to test atomics
__kernel void atomic_test (__global atomic_uint* T) {
      
    // Increment T atomically
    atomic_fetch_add(T, 1);

    // Make a fence to synchronise memory 
    // within the work item to global memory
    atomic_work_item_fence(
            CLK_GLOBAL_MEM_FENCE,
            memory_order_acq_rel,
            memory_scope_device);
}
