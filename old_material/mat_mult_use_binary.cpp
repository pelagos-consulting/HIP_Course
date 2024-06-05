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

    // We are going to do a simple array multiplication for this example, using raw binary files for input and output
    size_t nrows=1024;
    size_t ncols=1024;
    size_t element_size=sizeof(float);
    size_t nelements=nrows*ncols;
    size_t nbytes=nelements*element_size;

    // Allocate memory for the input and output arrays
    float* array_A_1D=(float*)malloc(nbytes);
    float* array_B_1D=(float*)malloc(nbytes);
    float* array_C_1D=(float*)malloc(nbytes);
    float* array_C_answer_1D=(float*)malloc(nbytes);

    // Read input data, this must be of size nrows*ncols*element_size, 
    // and the files array_A_1D.dat and array_B_1D.dat and array_C_answer_1D.dat must be in the current directory
    
    FILE* fp;
    // Read in matrix A
    fp=fopen("array_A_1D.dat","r");
    assert(fp!=NULL);
    fread(array_A_1D, element_size, nelements, fp);
    fclose(fp);

    // Read in matrix B
    fp=fopen("array_B_1D.dat","r");
    assert(fp!=NULL);
    fread(array_B_1D, element_size, nelements, fp);
    fclose(fp);

    // Read in the answer
    fp=fopen("array_C_answer_1D.dat","r");
    assert(fp!=NULL);
    fread(array_C_answer_1D, element_size, nelements, fp);
    fclose(fp); 

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

    // Make buffers for bringing data in and out of the computation
    cl_mem buffer_A=clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &errcode);
    errchk(errcode, "Creating buffer_A");
    cl_mem buffer_B=clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &errcode);
    errchk(errcode, "Creating buffer_B");
    cl_mem buffer_C=clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &errcode);
    errchk(errcode, "Creating buffer_C");

    // Load kernel 0 from file
    char filename[50];
    sprintf(filename, "kernels_device_%01d.bin",0);

    // Get the size of the file in a portable way
    struct stat stat_buf;
    errcode = stat(filename, &stat_buf);
    assert(errcode==0);
    const size_t file_bytes=stat_buf.st_size;

    // Create a character buffer 
    unsigned char* buffer=(unsigned char *)calloc(file_bytes, sizeof(unsigned char));

    // Read the binary code into the file
    fp=fopen(filename, "r");
    assert(fp!=NULL);

    // Read the file
    fread(buffer, sizeof(unsigned char), file_bytes, fp );
  
    // Close the file
    fclose(fp);

    const unsigned char* binary_source=buffer; 
    
    // Turn the binary code into a program
    cl_program program=clCreateProgramWithBinary(   context, 
                                                    1,
                                                    &device,
                                                    &file_bytes,
                                                    &binary_source,
                                                    NULL,
                                                    &errcode);

    errchk(errcode, "Creating program from source");

    // Free the binary source code 
    free(buffer);

    // Build the program
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

    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program,"mat_multiply",&errcode);
    errchk(errcode, "Creating Kernel");

    // Write memory to the buffer from the host device
    errchk(clEnqueueWriteBuffer(    command_queue,
                            buffer_A,
                            CL_TRUE,
                            0,
                            nbytes,
                            array_A_1D,
                            0,
                            NULL,
                            NULL), "Writing to buffer_A from host");

    errchk(clEnqueueWriteBuffer(    command_queue,
                            buffer_B,
                            CL_TRUE,
                            0,
                            nbytes,
                            array_B_1D,
                            0,
                            NULL,
                            NULL), "Writing to buffer_B from host");

    // Now run the kernel
    
    // Set arguments to the kernel
    errchk(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A ),"setting kernel argument 0");
    errchk(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B ),"setting kernel argument 1");
    errchk(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C ),"setting kernel argument 2");
    errchk(clSetKernelArg(kernel, 3, sizeof(int), &nrows ),"setting kernel argument 3");
    errchk(clSetKernelArg(kernel, 4, sizeof(int), &nrows ),"setting kernel argument 4");

    // Number of dimensions in the kernel
    cl_uint work_dim=2;
    const size_t global_work_size[]={ nrows, ncols };
    cl_event kernel_event;

    // Now enqueue the kernel
    errchk(clEnqueueNDRangeKernel(  command_queue,
                                    kernel,
                                    work_dim,
                                    NULL,
                                    global_work_size,
                                    NULL,
                                    0,
                                    NULL,
                                    &kernel_event), "Running the kernel");

    // Read memory from the buffer to the host
    errchk(clEnqueueReadBuffer(   command_queue,
                            buffer_C,
                            CL_TRUE,
                            0,
                            nbytes,
                            array_C_1D,
                            1,
                            &kernel_event,
                            NULL), "Copying matrix C from device to host");

    // Write out the computed answer to file
    fp=fopen("array_C_1D.dat","w");
    assert(fp!=NULL);
    fwrite(array_C_1D, element_size, nelements, fp);
    fclose(fp); 

    // Check the difference between the original and the computed matrix product
    // using the Root Mean Squared indicator
    double rms=0.0;
    for (int i=0; i<nelements; i++ ) {
        rms+=(array_C_1D[i]-array_C_answer_1D[i])*(array_C_1D[i]-array_C_answer_1D[i]);
    }
    rms/=nelements;
    rms=sqrt(rms);
    
    printf("RMS difference is %g\n", rms);

    // Wait for all command queues to finish
    // Release the command queues
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
    free(command_queues_1d);
    free(ndevices_1d);
    free(platformIDs_1d);
    free(contexts_1d);
    free(devices_2d);
    free(array_A_1D);
    free(array_B_1D);
    free(array_C_1D);

    // Stop the clock
    high_resolution_clock::time_point time2 = high_resolution_clock::now();
    duration<double> elapsed_time = duration_cast<duration<double>>(time2-time1);
    std::cout << "Elapsed time is " << elapsed_time.count() << "seconds" << std::endl;
}

