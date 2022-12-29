// Include the OpenCL helper headers
#include "cl_helper.hpp"

// Main program
int main(int argc, char** argv) {
    
    // Setting up memory for contexts and platforms
    cl_uint num_platforms, num_devices;
    
    cl_device_id *devices;
    cl_context *contexts;
    cl_platform_id *platforms;
    cl_int ret_code = CL_SUCCESS;
    
    // Could be CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_CPU
    cl_device_type device_type = CL_DEVICE_TYPE_ALL;
    
    // Get devices and contexts
    h_acquire_devices(device_type, 
                    &platforms, &num_platforms, 
                    &devices, &num_devices,
                    &contexts);
    
    // Report on available devices
    for (cl_uint n=0; n<num_devices; n++) {
        std::cout << "device " << n << std::endl;
        h_report_on_device(devices[n]);
    }
    
    // Release devices and contexts
    h_release_devices(devices, num_devices, contexts, platforms);
}