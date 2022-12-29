/*

MIT License

Copyright (c) 2018 Dr. Toby Potter and contributors from Pelagos Consulting and Education
Contact the author at tobympotter@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sys/stat.h>
#include <chrono>
#include <iostream>

#define OCL_EXIT -20
#define MAXCHAR 100
#define NQUEUES_PER_DEVICE 2

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.hpp"
#endif

#include "helper_functions.hpp"

int main(int argc, char**argv) {

    using namespace std::chrono;

    // Start the clock
    high_resolution_clock::time_point time1 = high_resolution_clock::now();

    // Useful for checking OpenCL errors
    cl_int errcode;

    // Number to hold the number of platforms
    cl_uint num_platforms;
    
    // First call this to get the number of compute platforms
    errchk(clGetPlatformIDs(0, NULL, &num_platforms),"Get number of platforms");
    
    // Allocate memory for the found number of platforms
    cl_platform_id* platformIDs_1d=(cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
    
    // Now fill in the allocated platform ID's
    errchk(clGetPlatformIDs(num_platforms, platformIDs_1d, &num_platforms),"Filling platform ID's\n");  

    // Now get information on each platform that was discovered
    for (int i=0; i<num_platforms; i++) {
        char version[MAXCHAR];
        char vendor[MAXCHAR];
        char name[MAXCHAR];
        
        // Fill the string with platform information
        errchk(clGetPlatformInfo(   platformIDs_1d[i], \
                                    CL_PLATFORM_NAME, 
                                    MAXCHAR, 
                                    name, 
                                    NULL),"Getting name\n");
        // Getting vendor         
        errchk(clGetPlatformInfo(platformIDs_1d[i], 
                                    CL_PLATFORM_VENDOR, 
                                    MAXCHAR, 
                                    vendor, 
                                    NULL),"Getting vendor\n");
        
        // Getting version
        errchk(clGetPlatformInfo(   platformIDs_1d[i], 
                                    CL_PLATFORM_VERSION,
                                    MAXCHAR,
                                    version,
                                    NULL),"Getting version\n");
                                    
        printf("Platform %d: %s, vendor: %s, version %s\n", i, name, vendor, version);
    }

    // Select the device we are going to use 
    cl_device_type target_device=CL_DEVICE_TYPE_ALL;

    // Allocate memory to hold the total number of devices per platform 
    cl_uint* ndevices_1d=(cl_uint*)calloc(num_platforms, sizeof(cl_uint));
    // Number of devices per platform
    int ndevices=0;
    // Number of platforms with devices that match what we want
    int num_valid_platforms=0;
    for (int i=0; i<num_platforms; i++) {
        cl_uint ndevice=0;

        // First pass, get the number of device per platform
        errcode=clGetDeviceIDs( platformIDs_1d[i],
                                target_device,
                                0,
                                NULL,
                                &ndevice);
        
        // Some OpenCL implementations yield an error if no devices of 
        // the right type exist, we must manually check
        if (errcode==-1) {
            // There are no devices
            ndevice=0;
        } else {
            errchk(errcode,"Getting number of devices");
        }
        
        if (ndevice>0) {
            // We have more than one device in this platform
            ndevices_1d[i]=ndevice;
            ndevices+=ndevice;
            num_valid_platforms++;
        } 

        printf("Platform %d has %d devices\n", i,  ndevices_1d[i]);
    }
    
    // Make sure we have at least one valid platform
    assert(num_valid_platforms>=1);

    // Create a 2D array for devices and fill it with device information for each platform
    // that has one or more devices in it
    cl_device_id** devices_2d=(cl_device_id**)calloc(num_valid_platforms, sizeof(cl_device_id*));
    int platform_counter=0;
    int device_counter=0;
    for (int i=0; i<num_platforms; i++) {
        // Skip over platforms with no devices in them 
        if (ndevices_1d[i]>0) {
            // Construct the devices array to hold the desired number of devices
            devices_2d[platform_counter]=(cl_device_id*)calloc(ndevices_1d[i], sizeof(cl_device_id));
        
            // Fill the devices array
            errchk(clGetDeviceIDs(  platformIDs_1d[i],
                                    target_device,
                                    ndevices_1d[i],
                                    devices_2d[platform_counter],
                                    NULL),"Filling device arrays");
                        // Loop over devices and get their name and global size information
            for (int j=0; j<ndevices_1d[i]; j++) {
                printf("Device %d:\n", device_counter);
                report_on_device(devices_2d[platform_counter][j]);
                device_counter++;
            }

            platform_counter++; 
        }
    }

    // Now make a context for each valid platform
    // Allocate memory to store the contexts
    cl_context* contexts_1d=(cl_context*)calloc(num_valid_platforms, sizeof(cl_context));
    platform_counter=0;
    for (int i=0; i<num_platforms; i++) {
        if (ndevices_1d[i]>0) {
            // We have a valid platform, create a context.
            // Handling the context properties is tricky, here is how to do it
            const cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, 
                                                  (cl_context_properties)platformIDs_1d[i], 
                                                  0 };

            // Now create a context using the platform and devices
            contexts_1d[platform_counter]=clCreateContext(prop, 
                                                          ndevices_1d[i], 
                                                          devices_2d[platform_counter], 
                                                          NULL, 
                                                          NULL, 
                                                          &errcode);

            errchk(errcode, "Creating contexts");
            platform_counter++;
        }
    }

    // Now create command queues for each valid device;
    int num_command_queues=ndevices*NQUEUES_PER_DEVICE;
    cl_command_queue* command_queues_1d=(cl_command_queue*)calloc(num_command_queues, sizeof(cl_command_queue));
    platform_counter=0;
    int queue_counter=0;
    for (int i=0; i<num_platforms; i++) {
        for (int j=0; j<ndevices_1d[i]; j++) {
            for (int k=0; k<NQUEUES_PER_DEVICE; k++) {
                command_queues_1d[queue_counter]=clCreateCommandQueue(  contexts_1d[platform_counter],
                                                                        devices_2d[platform_counter][j], 
                                                                        0, 
                                                                        &errcode);
                queue_counter++;
            }

        }
        if (ndevices_1d[i]>0) platform_counter++;
    }

    // Select a command queue to use from the pool of valid command queues
    cl_command_queue command_queue=command_queues_1d[0];
    
    // Get the context, device, and platform from the selected command queue
    cl_context context;
    cl_device_id device;

    errchk(clGetCommandQueueInfo(   command_queue, 
                                    CL_QUEUE_CONTEXT, 
                                    sizeof(context), 
                                    &context,
                                    NULL), "Getting the context");

    errchk(clGetCommandQueueInfo(   command_queue, 
                                    CL_QUEUE_DEVICE, 
                                    sizeof(device), 
                                    &device,
                                    NULL), "Getting the device");

    // Now specify the kernel source
    const char* kernel_source="__kernel void mat_multiply ( __global float* A, \n\
                                                            __global float* B, \n\
                                                            __global float* C, \n\
                                                            int nrows_A, \n\
                                                            int nrows_B) { \n\
                                                            \
        // nrows and ncols specify the dimensions of the output matrices \n\
        // Using Fortran ordering \n\
        size_t i0=get_global_id(0); \n\
        size_t i1=get_global_id(1); \n\
\n\
        // Loop over columns of A and rows of B \n\
        float temp=0.0; \n\
        for (int n=0; n<nrows_B; n++) { \
            // C has the same number of rows as A, and the same number of columns as B \n\
            temp+=A[n*nrows_A+i0]*B[i1*nrows_B+n]; \n\
        } \n\
        C[i1*nrows_A+i0]=temp; \n\
    }";

    // Turn this source code into a program
    cl_program program=clCreateProgramWithSource(   context, 
                                                    1, 
                                                    &kernel_source,
                                                    NULL,
                                                    &errcode);
    errchk(errcode, "Creating program from source");


    // Build the program for the device
    const char* build_opts="";
    errcode=clBuildProgram( program,
                            1,
                            &device,
                            build_opts,
                            NULL,
                            NULL);

    // Check the program build
    if (errcode!=CL_SUCCESS) {
        size_t elements;
        errchk(clGetProgramBuildInfo(   program,
                                        device,
                                        CL_PROGRAM_BUILD_LOG,
                                        0,
                                        NULL,
                                        &elements),"Checking build log");

        // Make up the build log string
        char* buildlog=(char*)calloc(elements, 1);

        errchk(clGetProgramBuildInfo(   program,
                                        device,
                                        CL_PROGRAM_BUILD_LOG,
                                        elements,
                                        buildlog,
                                        NULL), "Filling the build log");
        printf("Build log is %s\n", buildlog);
        exit(OCL_EXIT);
    }

    // Get the number of devices in the program, should be 1, we pretend there are many
    cl_uint nprogram_devices;
    errchk(clGetProgramInfo(    program, 
                                CL_PROGRAM_NUM_DEVICES, 
                                sizeof(cl_uint), 
                                &nprogram_devices, 
                                NULL),
                                "Getting the number of program devices");

    // Get the size of binary code for all devices in the program
    size_t binary_sizes[nprogram_devices];
    errchk(clGetProgramInfo(    program, 
                                CL_PROGRAM_BINARY_SIZES, 
                                sizeof(size_t)*nprogram_devices, 
                                binary_sizes,
                                NULL),
                                "Getting the size of compiled binaries");

    // Make an array for each binary created
    unsigned char* binary_codes[nprogram_devices];
    for (int n=0; n<nprogram_devices; n++) {
        binary_codes[n]=(unsigned char*)calloc(binary_sizes[n], sizeof(unsigned char));
    }

    // Fill the arrays with binary information
    errchk(clGetProgramInfo(    program,
                                CL_PROGRAM_BINARIES,
                                sizeof(unsigned char*)*nprogram_devices,
                                binary_codes,
                                NULL),
                                "Retrieving the compiled binaries");
    
    // Now save the compiled binaries to disk
    for (int n=0; n<nprogram_devices; n++) {
        char filename[50];
        sprintf(filename, "kernels_device_%01d.bin",n);
        FILE* fp=fopen(filename,"w");
        fwrite(binary_codes[n], binary_sizes[n], sizeof(unsigned char), fp);
        fclose(fp);
    }

    // Wait for all command queues to finish and release them
    for (int i=0; i<num_command_queues; i++) {
        errchk(clFinish(command_queues_1d[i]),"Finishing up command queues");
        errchk(clReleaseCommandQueue(command_queues_1d[i]), "Releasing command queues");
    }

    // Release the contexts and free the devices associated with each context
    for (int i=0; i<num_valid_platforms; i++) {
        errchk(clReleaseContext(contexts_1d[i]),"Releasing the context");
        free(devices_2d[i]);
    }

    // Clean up memory    
    for (int n=0; n<nprogram_devices; n++) {
        free(binary_codes[n]); 
    }
    
    free(command_queues_1d);
    free(ndevices_1d);
    free(platformIDs_1d);
    free(contexts_1d);
    free(devices_2d);

        // Stop the clock
    high_resolution_clock::time_point time2 = high_resolution_clock::now();
    duration<double> elapsed_time = duration_cast<duration<double>>(time2-time1);
    std::cout << "Elapsed time is " << elapsed_time.count() << "seconds" << std::endl;
}

