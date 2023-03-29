
# Include general environment variables
include ../env

# List of applications to target
TARGETS=hello_jobstep.exe hello_devices_mpi.exe hello_devices_mpi_onefile_hipcc.exe hello_devices_mpi_onefile_CC.exe

all: $(TARGETS)

# Compiler flags for C++ compiler
CC_CFLAGS=-g -fopenmp $(BASE_INC_FLAGS) -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --offload-arch=gfx90a -x hip

# Linker flags for C++ compiler
#CC_LDFLAGS=-g --rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64 $(BASE_LIB_FLAGS)
CC_LDFLAGS=$(BASE_LIB_FLAGS)

# Compiler flags for hipcc on AMD platform
HIPCC_CFLAGS=-ggdb -fopenmp $(BASE_INC_FLAGS) -I${MPICH_DIR}/include --offload-arch=gfx90a $(shell CC --cray-print-opts=cflags)

# Linker flags for hipcc on AMD platform
HIPCC_LDFLAGS=-L$(MPICH_DIR)/lib -lmpi $(PE_MPICH_GTL_DIR_amd_gfx90a) $(PE_MPICH_GTL_LIBS_amd_gfx90a) $(shell CC --cray-print-opts=libs) $(BASE_LIB_FLAGS)

# Compile the kernel source only with hipcc
%.o : %.hip.cpp
	hipcc -c $(HIPCC_CFLAGS) $< -o $@ 

# Compile the main source files with CC
%.o: %.cpp
	CC -c $(CC_CFLAGS) $< -o $@ 

# Link with hipcc
hello_devices_mpi.exe : kernels.o hello_devices_mpi.o
	hipcc $(HIPCC_CFLAGS) $^ -o $@ $(HIPCC_LDFLAGS)

# Alternatively, compile in one step with hipcc
hello_devices_mpi_onefile_hipcc.exe : hello_devices_mpi_onefile.cpp
	hipcc $(HIPCC_CFLAGS) $< -o $@ $(HIPCC_LDFLAGS)

# Alternatively, compile in one step with CC
hello_devices_mpi_onefile_CC.exe : hello_devices_mpi_onefile.cpp
	CC $(CC_CFLAGS) $< -o $@ $(CC_LDFLAGS)

# Compile the hello_jobstep helper application from Thomas Papatheodore
hello_jobstep.exe : hello_jobstep.cpp
	hipcc $(HIPCC_CFLAGS) $< -o $@ $(HIPCC_LDFLAGS)

# Clean step
clean:
	rm -rf *.exe
	rm -rf *.o


.EXPORT_ALL_VARIABLES:
