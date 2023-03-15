
# Include general environment variables
include ../env

# Location of general helper files
INC_DIR=../include

# List of applications to target
TARGETS=hello_devices_mpi.exe

all: $(TARGETS)

# Compile the kernels
%.o : %.hip.cpp
    hipcc -c $< -o $@ 

# Compile the main source file
%.o: *.cpp
    CC -c -std=c++11 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=$(ROCM_PATH) --offload-arch=gfx90a -x hip

# Linking step
%.exe : %.o
    hipcc -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
