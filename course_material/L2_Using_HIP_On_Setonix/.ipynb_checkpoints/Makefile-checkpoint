
# Include general environment variables
include ../env

# Location of general helper files
INC_DIR=../include

# List of applications to target
TARGETS=hello_devices_mpi.exe

all: $(TARGETS)

# Compiler flags for C++ compiler
CC_CFLAGS=-I$(INC_DIR) -std=c++11 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=$(ROCM_PATH) -fopenmp --offload-arch=gfx90a -x hip

# Linker flags for C++ compiler
CC_LDFLAGS=-fopenmp --rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64


# Compiler flags for hipcc on AMD platform
HIPCC_CFLAGS=-I$(INC_DIR) --offload-arch=gfx90a -ggdb -fopenmp

# Linker flags for hipcc on AMD platform
HIPCC_LDFLAGS=-fopenmp -L$(MPICH_DIR)/lib -lmpi $(PE_MPICH_GTL_DIR_amd_gfx90a) $(PE_MPICH_GTL_LIBS_amd_gfx90a)

# Compile the kernel source only with hipcc
%.o : %.hip.cpp
	hipcc -c $(HIPCC_CFLAGS) $< -o $@ 

# Compile the main source files with CC
%.o: %.cpp
	CC -c $(CC_CFLAGS) $< -o $@ 

# Link with hipcc
hello_devices_mpi.exe : kernels.o hello_devices_mpi.o
	hipcc $^ -o $@ $(HIPCC_LDFLAGS)

# Alternatively, compile in one step with hipcc
hello_devices_mpi_onefile_hipcc.exe : hello_devices_mpi_onefile.cpp
	hipcc $(HIPCC_CFLAGS) $< -o $@ $(HIPCC_LDFLAGS)

# Alternatively, compile in one step with CC
hello_devices_mpi_onefile_CC.exe : hello_devices_mpi_onefile.cpp
	CC $(CC_CFLAGS) $< -o $@ $(CC_LDFLAGS)


# Clean step
clean:
	rm -r *.exe
	rm -r *.o


.EXPORT_ALL_VARIABLES:
