#include <iostream>

void errchk(cl_int errcode, std::string message) {
    if (errcode!=CL_SUCCESS) {
        printf("Error, OpenCL call, failed at %s with error code %d \n", message.c_str(), errcode);
        exit(OCL_EXIT);
    }
};

void report_on_device(cl_device_id device) {
    using namespace std;

    // Report some information on the device
    size_t nbytes_name;
    errchk(clGetDeviceInfo(device, CL_DEVICE_NAME, NULL, NULL, &nbytes_name),"Device name bytes");
    char* name=new char[nbytes_name];
    errchk(clGetDeviceInfo(device, CL_DEVICE_NAME, nbytes_name, name, NULL),"Device name");
    int textwidth=16;

    printf("\t%20s %s \n","name:", name);


    cl_ulong mem_size;
    errchk(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size,
    NULL),"Global mem size");

    printf("\t%20s %d MB\n","global memory size:",mem_size/(1000000));

    errchk(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size,
    NULL),"Max mem alloc size");
   
    printf("\t%20s %d MB\n","max buffer size:", mem_size/(1000000));
    delete [] name;
}
