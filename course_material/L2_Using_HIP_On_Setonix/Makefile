
# List of applications to target
TARGETS=hello_jobstep.exe hello_devices_mpi.exe hello_devices_mpi_onefile_hipcc.exe hello_devices_mpi_onefile_CC.exe

all: $(TARGETS)

# Compiler flags for C++ compiler
CC_CFLAGS=-x hip -fPIC -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --offload-arch=gfx90a --rocm-path=${ROCM_PATH}

# Linker flags for C++ compiler
CC_LDFLAGS=--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64

# Compiler flags for hipcc on AMD platform
HIPCC_CFLAGS=-fPIC -fopenmp -I${MPICH_DIR}/include --offload-arch=gfx90a 

# Linker flags for hipcc on AMD platform
HIPCC_LDFLAGS=--offload-arch=gfx90a -L$(MPICH_DIR)/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa 

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
