/**********************************************************
"Hello World"-type program to test different srun layouts.

Written by Tom Papatheodore
**********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <iomanip>
#include <string.h>
#include <mpi.h>
#include <sched.h>
#include <hip/hip_runtime.h>
#include <omp.h>

// Macro for checking errors in HIP API calls
#define hipErrorCheck(call)                                                                 \
do{                                                                                         \
    hipError_t hipErr = call;                                                               \
    if(hipSuccess != hipErr){                                                               \
        printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErr)); \
        exit(0);                                                                            \
    }                                                                                       \
}while(0)

int main(int argc, char *argv[]){

	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char name[MPI_MAX_PROCESSOR_NAME];
	int resultlength;
	MPI_Get_processor_name(name, &resultlength);

    // If ROCR_VISIBLE_DEVICES is set, capture visible GPUs
    const char* gpu_id_list; 
    const char* rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
    if(rocr_visible_devices == NULL){
        gpu_id_list = "N/A";
    }
    else{
        gpu_id_list = rocr_visible_devices;
    }

	// Find how many GPUs HIP runtime says are available
	int num_devices = 0;
    hipErrorCheck( hipGetDeviceCount(&num_devices) );

	int hwthread;
	int thread_id = 0;

	if(num_devices == 0){
		#pragma omp parallel default(shared) private(hwthread, thread_id)
		{
			thread_id = omp_get_thread_num();
			hwthread = sched_getcpu();

            printf("MPI %03d - OMP %03d - HWT %03d - Node %s\n", 
                    rank, thread_id, hwthread, name);

		}
	}
	else{

		char busid[64];

        std::string busid_list = "";
        std::string rt_gpu_id_list = "";

		// Loop over the GPUs available to each MPI rank
		for(int i=0; i<num_devices; i++){

			hipErrorCheck( hipSetDevice(i) );

			// Get the PCIBusId for each GPU and use it to query for UUID
			hipErrorCheck( hipDeviceGetPCIBusId(busid, 64, i) );

			// Concatenate per-MPIrank GPU info into strings for print
            if(i > 0) rt_gpu_id_list.append(",");
            rt_gpu_id_list.append(std::to_string(i));

            std::string temp_busid(busid);

            if(i > 0) busid_list.append(",");
            busid_list.append(temp_busid.substr(5,2));            

		}

		#pragma omp parallel default(shared) private(hwthread, thread_id)
		{
            #pragma omp critical
            {
			thread_id = omp_get_thread_num();
			hwthread = sched_getcpu();

            printf("MPI %03d - OMP %03d - HWT %03d - Node %s - RT_GPU_ID %s - GPU_ID %s - Bus_ID %s\n",
                    rank, thread_id, hwthread, name, rt_gpu_id_list.c_str(), gpu_id_list, busid_list.c_str());
           }
		}
	}

	MPI_Finalize();

	return 0;
}

