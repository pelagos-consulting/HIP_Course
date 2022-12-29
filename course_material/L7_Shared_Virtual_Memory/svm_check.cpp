/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <iostream>

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

int main(int argc, char** argv) {
    
    // Parse arguments and set the target device
    cl_device_type target_device;
    cl_uint dev_index = h_parse_args(argc, argv, &target_device);
    
    // Useful for checking OpenCL errors
    cl_int errcode;

    // Create handles to platforms, 
    // devices, and contexts

    // Number of platforms discovered
    cl_uint num_platforms;

    // Number of devices discovered
    cl_uint num_devices;

    // Pointer to an array of platforms
    cl_platform_id *platforms = NULL;

    // Pointer to an array of devices
    cl_device_id *devices = NULL;

    // Pointer to an array of contexts
    cl_context *contexts = NULL;
    
    // Helper function to acquire devices
    h_acquire_devices(target_device,
                     &platforms,
                     &num_platforms,
                     &devices,
                     &num_devices,
                     &contexts);
    
    // Number of command queues to generate
    cl_uint num_command_queues = num_devices;
    
    // Make sure command line arguments are sane
    assert(dev_index < num_devices);
    cl_context context = contexts[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // Check if the device supports coarse-grained SVM
    cl_device_svm_capabilities svm;
    errcode = clGetDeviceInfo(
        device,
        CL_DEVICE_SVM_CAPABILITIES,
        sizeof(cl_device_svm_capabilities),
        &svm,
        NULL
    );
    
    if (errcode == CL_INVALID_VALUE) { 
        printf("Sorry, this device does not support Shared Virtual Memory.");
        exit(OCL_EXIT);
    }    
    
    if (errcode == CL_SUCCESS && 
        (svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
        printf("Device supports coarse-grained SVM\n");
    }
        
    if (errcode == CL_SUCCESS && 
        (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
        printf("Device supports fine-grained buffer SVM\n");
    }
 
    if (errcode == CL_SUCCESS && 
        (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) &&
        (svm & CL_DEVICE_SVM_ATOMICS)) {
        printf("Device supports fine-grained buffer SVM with atomics\n");
    }
    
    if (errcode == CL_SUCCESS && 
        (svm & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
        printf("Device supports fine-grained system SVM\n");
    }
 
    if (errcode == CL_SUCCESS && 
        (svm & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) && 
        (svm & CL_DEVICE_SVM_ATOMICS)) {
        printf("Device supports fine-grained buffer SVM with atomics\n");
    }
             
    // Clean up devices, queues, and contexts
    h_release_devices(
        devices,
        num_devices,
        contexts,
        platforms
    );
}

