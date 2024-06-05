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
                                                                        CL_QUEUE_PROFILING_ENABLE, 
                                                                        &errcode);
                queue_counter++;
            }

        }
        if (ndevices_1d[i]>0) platform_counter++;
    }

    // We are going to do a simple array multiplication for this example, using raw binary files for input and output
    size_t nrows_A=1024;
    size_t ncols_A=1024;

    size_t nrows_B=1024;
    size_t ncols_B=1024;

    size_t nrows_C=nrows_A;
    size_t ncols_C=ncols_B;

    // Dimensions of A transposed
    size_t nrows_A_transp=ncols_A;
    size_t ncols_A_transp=nrows_A;

    size_t element_size=sizeof(float);
    size_t nelements_A=nrows_A*ncols_A;
    size_t nelements_B=nrows_B*ncols_B;
    size_t nelements_C=nrows_C*ncols_C;

    // Number of bytes in each matrix
    size_t nbytes_A=nelements_A*element_size;
    size_t nbytes_B=nelements_B*element_size;
    size_t nbytes_C=nelements_C*element_size;

    // Allocate memory for the input and output arrays
    float* array_A_1D=(float*)malloc(nbytes_A);
    float* array_B_1D=(float*)malloc(nbytes_B);
    float* array_C_1D=(float*)malloc(nbytes_C);
    float* array_C_answer_1D=(float*)malloc(nbytes_C);

    // Read input data, this must be of size nrows*ncols*element_size, 
    // and the files array_A_1D.dat and array_B_1D.dat and array_C_answer_1D.dat must be in the current directory
    
    FILE* fp;
    // Read in matrix A
    fp=fopen("array_A_1D.dat","r");
    assert(fp!=NULL);
    fread(array_A_1D, element_size, nelements_A, fp);
    fclose(fp);

    // Read in matrix B
    fp=fopen("array_B_1D.dat","r");
    assert(fp!=NULL);
    fread(array_B_1D, element_size, nelements_B, fp);
    fclose(fp);

    // Read in the answer
    fp=fopen("array_C_answer_1D.dat","r");
    assert(fp!=NULL);
    fread(array_C_answer_1D, element_size, nelements_C, fp);
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
    cl_mem buffer_A=clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes_A, NULL, &errcode);
    errchk(errcode, "Creating buffer_A");
    // Make a buffer for transposed A, then B and C
    cl_mem buffer_A_transp=clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes_A, NULL, &errcode);
    errchk(errcode, "Creating buffer_A_transp");
    cl_mem buffer_B=clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes_B, NULL, &errcode);
    errchk(errcode, "Creating buffer_B");
    cl_mem buffer_C=clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes_C, NULL, &errcode);
    errchk(errcode, "Creating buffer_C");

    // Now specify the source code for all the kernels in use
    
    const char* kernel_source="\n\
        // kernel to do a matrix transpose \n\
        __kernel void mat_transpose(    __global float* src, \n\
                                        __global float* dest, \n\
                                        int nrows_src, \n\
                                        int nrows_dest) { \n\
            // We assume Fortran ordering for the matrices \n\
            // i0, and i1 represent the coordinates of src \n\
            // coordinates are reversed for dest \n\
            size_t i0=get_global_id(0); \n\
            size_t i1=get_global_id(1); \n\
            size_t offset_src=i1*nrows_src+i0; \n\
            size_t offset_dest=i0*nrows_dest+i1; \n\
            dest[offset_dest]=src[offset_src]; \n\
        } \n\
        \n\
        // standard matrix multiply kernel \n\
        __kernel void mat_mult (    __global float* A, \n\
                                    __global float* B, \n\
                                    __global float* C, \n\
                                    int nrows_A, \n\
                                    int nrows_B) { \n\
            \n\
            // i0 and i1 represent the coordinates in C \n\
            // We assume Fortran ordering for the matrices \n\
            size_t i0=get_global_id(0); \n\
            size_t i1=get_global_id(1); \n\
            size_t offset_B=i1*nrows_B; \n\
            float temp=0.0; \n\
            // Loop over columns of A and rows of B \n\
            for (int n=0; n<nrows_B; n++) { \
                // C has the same number of rows as A, and the same number of columns as B \n\
                // i0 is the row index of A \n\
                // i1 is the column index of B \n\
                temp+=A[n*nrows_A+i0]*B[offset_B+n]; \n\
            } \n\
            // Number of rows in C is same as number of rows in A \n\
            C[i1*nrows_A+i0]=temp; \n\
        } \n\
        \n\
        // special matrix multiply kernel that uses a pre-transposed matrix A\n\
        __kernel void mat_mult_transp ( __global float* A_transp, \n\
                                        __global float* B, \n\
                                        __global float* C, \n\
                                        int nrows_A_transp, \n\
                                        int nrows_B, \n\
                                        int nrows_C) { \n\
            // i0 and i1 represent the coordinates in C \n\
            // We assume Fortran ordering for the matrices \n\
            size_t i0=get_global_id(0); \n\
            size_t i1=get_global_id(1); \n\
            size_t offset_A=i0*nrows_A_transp; \n\
            size_t offset_B=i1*nrows_B; \n\
            float temp=0.0; \n\
            // For every coordinate in C, loop over the related rows of A_transp and B \n\
            for (int n=0; n<nrows_B; n++) { \
                // Every column of A_transp corresponds to a row of C \n\
                // Every column of B corresponds to a column of C \n\
                // C has the same number of rows as A_transp, and the same number of columns as B \n\
                // i0 is the column index of A_transp \n\
                // i1 is the column index of B \n\
                temp+=A_transp[offset_A+n]*B[offset_B+n]; \n\
            } \n\
            C[i1*nrows_C+i0]=temp; \n\
        } \n\
\n\
        // special matrix multiply kernel that uses a pre-transposed matrix and vectors A\n\
        __kernel void mat_mult_transp_vector ( __global float8* A_transp, \n\
                                        __global float8* B, \n\
                                        __global float* C, \n\
                                        int nrows_A_transp, \n\
                                        int nrows_B, \n\
                                        int nrows_C) { \n\
            // i0 and i1 represent the coordinates in C \n\
            // We assume Fortran ordering for the matrices \n\
            size_t i0=get_global_id(0); \n\
            size_t i1=get_global_id(1); \n\
            size_t offset_A=i0*nrows_A_transp; \n\
            size_t offset_B=i1*nrows_B; \n\
            float8 temp=0.0; \n\
            // For every coordinate in C, loop over the related rows of A_transp and B \n\
            for (int n=0; n<nrows_B; n++) { \
                // Every column of A_transp corresponds to a row of C \n\
                // Every column of B corresponds to a column of C \n\
                // C has the same number of rows as A_transp, and the same number of columns as B \n\
                // i0 is the column index of A_transp \n\
                // i1 is the column index of B \n\
                temp+=A_transp[offset_A+n]*B[offset_B+n]; \n\
            } \n\
            // Access components of a vector \n\
            C[i1*nrows_C+i0]=temp.s0+temp.s1+temp.s2+temp.s3+temp.s4+temp.s5+temp.s6+temp.s7; \n\
        } \n\
    ";

    // Turn the source code into a program
    cl_program program=clCreateProgramWithSource(   context, 
                                                    1, 
                                                    &kernel_source,
                                                    NULL,
                                                    &errcode);
    errchk(errcode, "Creating program from source");

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

    // Create kernels from the built program
    cl_kernel kernel_mat_transpose=clCreateKernel(program,"mat_transpose",&errcode);
    errchk(errcode, "Creating Kernel mat_transpose");

    cl_kernel kernel_mat_mult=clCreateKernel(program,"mat_mult",&errcode);
    errchk(errcode, "Creating Kernel mat_mult");

    cl_kernel kernel_mat_mult_transp=clCreateKernel(program,"mat_mult_transp",&errcode);
    errchk(errcode, "Creating Kernel mat_mul_transp");

    cl_kernel kernel_mat_mult_transp_vector=clCreateKernel(program,"mat_mult_transp_vector",&errcode);
    errchk(errcode, "Creating Kernel mat_mul_transp_vector");

    // Write memory to the buffer from the host device
    errchk(clEnqueueWriteBuffer(    command_queue,
                            buffer_A,
                            CL_TRUE,
                            0,
                            nbytes_A,
                            array_A_1D,
                            0,
                            NULL,
                            NULL), "Writing to buffer_A from host");

    errchk(clEnqueueWriteBuffer(    command_queue,
                            buffer_B,
                            CL_TRUE,
                            0,
                            nbytes_B,
                            array_B_1D,
                            0,
                            NULL,
                            NULL), "Writing to buffer_B from host");
    
    // Now run the kernels for many iterations
    int iterations=1;

    for (int n=0; n<iterations; n++) {

        // Set arguments to the transpose kernel
        errchk(clSetKernelArg(kernel_mat_transpose, 0, sizeof(cl_mem), &buffer_A ),"setting mat_transpose argument 0");
        errchk(clSetKernelArg(kernel_mat_transpose, 1, sizeof(cl_mem), &buffer_A_transp ),"setting mat_transpose argument 1");
        errchk(clSetKernelArg(kernel_mat_transpose, 2, sizeof(int), &nrows_A ),"setting mat_transpose argument 2");
        errchk(clSetKernelArg(kernel_mat_transpose, 3, sizeof(int), &nrows_A_transp ),"setting mat_transpose argument 3");
        
        // Set work size
        cl_uint work_dim=2;
        const size_t global_size_mat_transpose[]={ nrows_A, ncols_A };
        cl_event event_mat_transpose;

        // Now enqueue the transpose kernel
        errchk(clEnqueueNDRangeKernel(  command_queue,
                                        kernel_mat_transpose,
                                        work_dim,
                                        NULL,
                                        global_size_mat_transpose,
                                        NULL,
                                        0,
                                        NULL,
                                        &event_mat_transpose), "Running the transpose kernel");

        // Set arguments for the multiply kernel
        errchk(clSetKernelArg(kernel_mat_mult, 0, sizeof(cl_mem), &buffer_A ),"setting mat_mult argument 0");
        errchk(clSetKernelArg(kernel_mat_mult, 1, sizeof(cl_mem), &buffer_B ),"setting mat_mult argument 1");
        errchk(clSetKernelArg(kernel_mat_mult, 2, sizeof(cl_mem), &buffer_C ),"setting mat_mult argument 2");
        errchk(clSetKernelArg(kernel_mat_mult, 3, sizeof(int), &nrows_A ),"setting mat_mult argument 3");
        errchk(clSetKernelArg(kernel_mat_mult, 4, sizeof(int), &nrows_B ),"setting mat_mult argument 4");

        // Number of dimensions in the kernel
        const size_t global_size_mat_mult[]={ nrows_C, ncols_C };
        cl_event event_mat_mult;

        // Now enqueue the standard matrix multiply kernel
        errchk(clEnqueueNDRangeKernel(  command_queue,
                                        kernel_mat_mult,
                                        work_dim,
                                        NULL,
                                        global_size_mat_mult,
                                        NULL,
                                        1,
                                        &event_mat_transpose,
                                        &event_mat_mult), "Running the kernel");

        // Set arguments for the multiply kernel with transpose
        errchk(clSetKernelArg(kernel_mat_mult_transp, 0, sizeof(cl_mem), &buffer_A_transp ),"setting mat_mult_transp argument 0");
        errchk(clSetKernelArg(kernel_mat_mult_transp, 1, sizeof(cl_mem), &buffer_B ),"setting kernel mat_mult_transp argument 1");
        errchk(clSetKernelArg(kernel_mat_mult_transp, 2, sizeof(cl_mem), &buffer_C ),"setting kernel mat_mult_transp argument 2");
        errchk(clSetKernelArg(kernel_mat_mult_transp, 3, sizeof(int), &nrows_A_transp ),"setting mat_mult_transp argument 3");
        errchk(clSetKernelArg(kernel_mat_mult_transp, 4, sizeof(int), &nrows_B ),"setting mat_mult_transp argument 4");
        errchk(clSetKernelArg(kernel_mat_mult_transp, 5, sizeof(int), &nrows_C ),"setting mat_mult_transp argument 5");

        cl_event event_mat_mult_transp;

        // Now enqueue the kernel
        errchk(clEnqueueNDRangeKernel(  command_queue,
                                        kernel_mat_mult_transp,
                                        work_dim,
                                        NULL,
                                        global_size_mat_mult,
                                        NULL,
                                        1,
                                        &event_mat_mult,
                                        &event_mat_mult_transp), "Running the kernel");

        // The number of elements we are processing at a time
        cl_int vector_length=8;
        cl_int vectorised_nrows_A_transp=nrows_A_transp/vector_length;
        cl_int vectorised_nrows_B=nrows_B/vector_length;

        // Set arguments for the multiply kernel with transpose
        errchk(clSetKernelArg(kernel_mat_mult_transp_vector, 0, sizeof(cl_mem), &buffer_A_transp ),"setting \
        mat_mult_transp_vector argument 0");
        errchk(clSetKernelArg(kernel_mat_mult_transp_vector, 1, sizeof(cl_mem), &buffer_B ),"setting kernel \
        mat_mult_transp_vector argument 1");
        errchk(clSetKernelArg(kernel_mat_mult_transp_vector, 2, sizeof(cl_mem), &buffer_C ),"setting kernel \
        mat_mult_transp_vector argument 2");
        errchk(clSetKernelArg(kernel_mat_mult_transp_vector, 3, sizeof(int), &vectorised_nrows_A_transp ),"setting \
        mat_mult_transp_vector argument 3");
        errchk(clSetKernelArg(kernel_mat_mult_transp_vector, 4, sizeof(int), &vectorised_nrows_B ),"setting \
        mat_mult_transp_vector argument 4");
        errchk(clSetKernelArg(kernel_mat_mult_transp_vector, 5, sizeof(int), &nrows_C ),"setting \
        mat_mult_transp_vector argument 5");

        cl_event event_mat_mult_transp_vector;

        // Now enqueue the transposed and vectorised kernel
        errchk(clEnqueueNDRangeKernel(  command_queue,
                                        kernel_mat_mult_transp_vector,
                                        work_dim,
                                        NULL,
                                        global_size_mat_mult,
                                        NULL,
                                        1,
                                        &event_mat_mult_transp,
                                        &event_mat_mult_transp_vector), "Running the kernel");

        // Wait for all events to finish
        clFinish(command_queue);

        // Get the timing information from each event
        cl_ulong start_counter=0, end_counter=0;

        // Firstly the matrix transpose
        clGetEventProfilingInfo(    event_mat_transpose,
                                    CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),
                                    &start_counter,
                                    NULL);
        clGetEventProfilingInfo(    event_mat_transpose,
                                    CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),
                                    &end_counter,
                                    NULL);
        // This should give the time in milliseconds
        cl_double time_mat_transpose=(cl_double)(end_counter-start_counter)*(cl_double)1.0e-6;

        // next the standard matrix multiply
        clGetEventProfilingInfo(    event_mat_mult,
                                    CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),
                                    &start_counter,
                                    NULL);
        clGetEventProfilingInfo(    event_mat_mult,
                                    CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),
                                    &end_counter,
                                    NULL);
        // This should give the time in milliseconds
        cl_double time_mat_mult=(cl_double)(end_counter-start_counter)*(cl_double)1.0e-6;

        // Finally the transposed matrix multiply
        clGetEventProfilingInfo(    event_mat_mult_transp,
                                    CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),
                                    &start_counter,
                                    NULL);
        clGetEventProfilingInfo(    event_mat_mult_transp,
                                    CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),
                                    &end_counter,
                                    NULL);
        // This should give the time in milliseconds
        cl_double time_mat_mult_transp=(cl_double)(end_counter-start_counter)*(cl_double)1.0e-6;

        // Finally the transposed and vectorised matrix multiply
        clGetEventProfilingInfo(    event_mat_mult_transp_vector,
                                    CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),
                                    &start_counter,
                                    NULL);
        clGetEventProfilingInfo(    event_mat_mult_transp_vector,
                                    CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),
                                    &end_counter,
                                    NULL);
        // This should give the time in milliseconds
        cl_double time_mat_mult_transp_vector=(cl_double)(end_counter-start_counter)*(cl_double)1.0e-6;

        printf("Matrix transpose took %f ms\n", time_mat_transpose);
        printf("Standard matrix multiply took %f ms\n", time_mat_mult);
        printf("Transposed matrix multiply took %f ms\n", time_mat_mult_transp);
        printf("Transposed and vectorised matrix multiply took %f ms\n", time_mat_mult_transp_vector);
        printf("Transposed approach resulted in a speedup of %fx\n", time_mat_mult/(time_mat_transpose+time_mat_mult_transp));
        printf("Transposed and vectorised approach resulted in a speedup of %fx\n",
        time_mat_mult/(time_mat_transpose+time_mat_mult_transp_vector));
    }
    
    // Make sure all transfers are complete
    clFinish(command_queue);

    // Read memory from the buffer to the host
    errchk(clEnqueueReadBuffer(   command_queue,
                            buffer_C,
                            CL_TRUE,
                            0,
                            nbytes_C,
                            array_C_1D,
                            0,
                            NULL,
                            NULL), "Copying matrix C from device to host");

    // Write out the computed answer to file
    fp=fopen("array_C_1D.dat","w");
    assert(fp!=NULL);
    fwrite(array_C_1D, element_size, nelements_C, fp);
    fclose(fp); 

    // Check the difference between the original and the computed matrix product
    // using the Root Mean Squared indicator
    double rms=0.0;
    for (int i=0; i<nelements_C; i++ ) {
        rms+=(array_C_1D[i]-array_C_answer_1D[i])*(array_C_1D[i]-array_C_answer_1D[i]);
    }
    rms/=nelements_C;
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

