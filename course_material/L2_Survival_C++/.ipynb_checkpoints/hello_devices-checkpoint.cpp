#include <iostream>
#include <map>
#include <cstdio>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.hpp"
#endif

#define OCL_EXIT -20

// Lookup table for error codes
std::map<cl_int, std::string> error_codes {
    {CL_SUCCESS, "CL_SUCCESS"},
    {CL_BUILD_PROGRAM_FAILURE, "CL_BUILD_PROGRAM_FAILURE"},
    {CL_COMPILE_PROGRAM_FAILURE, "CL_COMPILE_PROGRAM_FAILURE"},
    {CL_COMPILER_NOT_AVAILABLE, "CL_COMPILER_NOT_AVAILABLE"},
    {CL_DEVICE_NOT_FOUND, "CL_DEVICE_NOT_FOUND"},
    {CL_DEVICE_NOT_AVAILABLE, "CL_DEVICE_NOT_AVAILABLE"},
    {CL_DEVICE_PARTITION_FAILED, "CL_DEVICE_PARTITION_FAILED"},
    {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {CL_IMAGE_FORMAT_MISMATCH, "CL_IMAGE_FORMAT_MISMATCH"},
    {CL_IMAGE_FORMAT_NOT_SUPPORTED, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {CL_INVALID_ARG_INDEX, "CL_INVALID_ARG_INDEX"},
    {CL_INVALID_ARG_SIZE, "CL_INVALID_ARG_SIZE"},
    {CL_INVALID_ARG_VALUE, "CL_INVALID_ARG_VALUE"},
    {CL_INVALID_BINARY, "CL_INVALID_BINARY"},
    {CL_INVALID_BUFFER_SIZE, "CL_INVALID_BUFFER_SIZE"},
    {CL_INVALID_BUILD_OPTIONS, "CL_INVALID_BUILD_OPTIONS"},
    {CL_INVALID_COMMAND_QUEUE, "CL_INVALID_COMMAND_QUEUE"},
    {CL_INVALID_COMPILER_OPTIONS, "CL_INVALID_COMPILER_OPTIONS"},
    {CL_INVALID_CONTEXT, "CL_INVALID_CONTEXT"},
    {CL_INVALID_DEVICE, "CL_INVALID_DEVICE"},
    {CL_INVALID_DEVICE_PARTITION_COUNT, "CL_INVALID_DEVICE_PARTITION_COUNT"},
    {CL_INVALID_DEVICE_TYPE, "CL_INVALID_DEVICE_TYPE"},
    {CL_INVALID_EVENT, "CL_INVALID_EVENT"},
    {CL_INVALID_EVENT_WAIT_LIST, "CL_INVALID_WAIT_LIST"},
    {CL_INVALID_GLOBAL_OFFSET, "CL_INVALID_GLOBAL_OFFSET"},
    {CL_INVALID_GLOBAL_WORK_SIZE, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {CL_INVALID_HOST_PTR, "CL_INVALID_HOST_PTR"},
    {CL_INVALID_IMAGE_DESCRIPTOR, "CL_INVALID_IMAGE_DESCRIPTOR"},
    {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {CL_INVALID_IMAGE_SIZE, "CL_INVALID_IMAGE_SIZE"},
    {CL_INVALID_KERNEL, "CL_INVALID_KERNEL"},
    {CL_INVALID_KERNEL_ARGS, "CL_INVALID_KERNEL_ARGS"},
    {CL_INVALID_KERNEL_DEFINITION, "CL_INVALID_KERNEL_DEFINITION"},
    {CL_INVALID_KERNEL_NAME, "CL_INVALID_KERNEL_NAME"},
    {CL_INVALID_LINKER_OPTIONS, "CL_INVALID_LINKER_OPTIONS"},
    {CL_INVALID_MEM_OBJECT, "CL_INVALID_MEM_OBJECT"},
    {CL_INVALID_OPERATION, "CL_INVALID_OPERATION"},
    {CL_INVALID_PLATFORM, "CL_INVALID_PLATFORM"},
    {CL_INVALID_PROGRAM, "CL_INVALID_PROGRAM"},
    {CL_INVALID_PROGRAM_EXECUTABLE, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {CL_INVALID_PROPERTY, "CL_INVALID_PROPERTY"},
    {CL_INVALID_QUEUE_PROPERTIES, "CL_INVALID_QUEUE_PROPERTIES"},
    {CL_INVALID_SAMPLER, "CL_INVALID_SAMPLER"},
    {CL_INVALID_VALUE, "CL_INVALID_VALUE"},
    {CL_INVALID_WORK_DIMENSION, "CL_INVALID_WORK_DIMENSION"},
    {CL_INVALID_WORK_GROUP_SIZE, "CL_INVALID_WORK_GROUP_SIZE"},
    {CL_INVALID_WORK_ITEM_SIZE, "CL_INVALID_WORK_ITEM_SIZE"},
    {CL_KERNEL_ARG_INFO_NOT_AVAILABLE, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
    {CL_LINK_PROGRAM_FAILURE, "CL_LINK_PROGRAM_FAILURE"},
    {CL_LINKER_NOT_AVAILABLE, "CL_LINKER_NOT_AVAILABLE"},
    {CL_MAP_FAILURE, "CL_MAP_FAILURE"},
    {CL_MEM_COPY_OVERLAP, "CL_MEM_COPY_OVERLAP"},
    {CL_MEM_OBJECT_ALLOCATION_FAILURE, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {CL_MISALIGNED_SUB_BUFFER_OFFSET, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {CL_OUT_OF_HOST_MEMORY, "CL_OUT_OF_HOST_MEMORY"},
    {CL_OUT_OF_RESOURCES, "CL_OUT_OF_RESOURCES"},
    {CL_PROFILING_INFO_NOT_AVAILABLE, "CL_PROFILING_INFO_NOT_AVAILABLE"},
};

// Function to check error code
void h_errchk(cl_int errcode, std::string message) {
    if (errcode!=CL_SUCCESS) {
        if (error_codes.count(errcode)>0) {
            printf("Error, Opencl call failed at \"%s\" with error code %s (%d)\n", 
                    message.c_str(), error_codes[errcode].c_str(), errcode);
        } else {
            printf("Error, OpenCL call failed at \"%s\" with error code %d\n", 
                    message.c_str(), errcode);
        }
        exit(OCL_EXIT);
    }
};

// Function to report information on a compute device
void h_report_on_device(cl_device_id device) {
    using namespace std;

    // Report some information on the device
    size_t nbytes_name;
    h_errchk(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &nbytes_name),"Device name bytes");
    char* name=new char[nbytes_name];
    h_errchk(clGetDeviceInfo(device, CL_DEVICE_NAME, nbytes_name, name, NULL),"Device name");
    int textwidth=16;

    printf("\t%20s %s \n","name:", name);

    cl_ulong mem_size;
    h_errchk(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size,
    NULL),"Global mem size");

    printf("\t%20s %d MB\n","global memory size:",mem_size/(1000000));

    h_errchk(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size,
    NULL),"Max mem alloc size");
   
    printf("\t%20s %d MB\n","max buffer size:", mem_size/(1000000));
    delete [] name;
}

// Function to create lists of contexts and devices that map to available hardware
void h_acquire_devices(
        // Input parameter
        cl_device_type device_type,
        // Output parameters
        cl_platform_id **platform_ids_out,
        cl_uint *num_platforms_out,
        cl_device_id **device_ids_out,
        cl_uint *num_devices_out, 
        cl_context **contexts_out) {

    // Return code for running things
    cl_int ret_code = CL_SUCCESS;
    
    // Get platforms
    cl_uint num_platforms;
    cl_platform_id *platform_ids = NULL;
    h_errchk(clGetPlatformIDs(0, NULL, &num_platforms), "Fetching number of platforms");
    platform_ids = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    h_errchk(clGetPlatformIDs(num_platforms, platform_ids, NULL), "Fetching platforms");
        
    // Fetch the total number of compatible devices
    cl_uint num_devices=0;
    
    // Get the total number of devices
    for (cl_uint n=0; n < num_platforms; n++) {

        cl_uint ndevices;
        ret_code = clGetDeviceIDs(
            platform_ids[n],
            device_type,
            0,
            NULL,
            &ndevices);

        if (ret_code != CL_DEVICE_NOT_FOUND) {
            h_errchk(ret_code, "Getting number of devices");
            num_devices += ndevices;
        }
    }
    
    if (num_devices == 0) {
        printf("Failed to find a suitable compute device\n");
        exit(OCL_EXIT);
    }

    // Allocate memory for device ID's and contexts
    cl_device_id *device_ids = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
    cl_context *contexts = (cl_context*)calloc(num_devices, sizeof(cl_context));
    
    cl_device_id *device_ids_ptr = device_ids;
    cl_context *contexts_ptr = contexts;
    
    // Fill device ID's array
    for (cl_uint n=0; n < num_platforms; n++) {
        cl_uint ndevices;

        ret_code = clGetDeviceIDs(
            platform_ids[n],
            device_type,
            0,
            NULL,
            &ndevices);

        if (ret_code != CL_DEVICE_NOT_FOUND) {
            h_errchk(ret_code, "Getting number of devices for the platform");
            
            // Fill devices
            h_errchk(clGetDeviceIDs(
                platform_ids[n],
                device_type,
                ndevices,
                device_ids_ptr,
                NULL), "Filling devices");
            
            // Create a context for every device found
            for (cl_uint c=0; c<ndevices; c++ ) {
                // Context properties
                const cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, 
                                                      (cl_context_properties)platform_ids[n], 
                                                      0 };
                
                // Create a context with 1 device in it
                const cl_device_id dev_id = *(device_ids_ptr+c);
                cl_uint ndev = 1;
                
                *contexts_ptr = clCreateContext(
                    prop, 
                    ndev, 
                    &dev_id,
                    NULL,
                    NULL,
                    &ret_code
                );
                h_errchk(ret_code, "Creating a context");
                contexts_ptr++;
            }
            
            // Advance device_id's pointer
            device_ids_ptr += ndevices;
        }
    }   

    // Fill in output information here to 
    // avoid problems with understanding
    *platform_ids_out = platform_ids;
    *num_platforms_out = num_platforms;
    *device_ids_out = device_ids;
    *num_devices_out = num_devices;
    *contexts_out = contexts;
}

// Function to release devices and contexts
void h_release_devices(
        cl_device_id *devices,
        cl_uint num_devices,
        cl_context* contexts,
        cl_platform_id *platforms) {
    
    // Free contexts
    for (cl_uint n = 0; n<num_devices; n++) {
        h_errchk(clReleaseContext(contexts[n]), "Releasing contexts");
    }

    free(contexts);
    free(devices);
    free(platforms);
}

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