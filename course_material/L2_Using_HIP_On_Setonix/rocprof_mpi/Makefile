
# List of applications to target
TARGETS=mat_mult_profiling_mpi.exe

all: $(TARGETS)

# Compiler flags for hipcc on AMD platform
HIPCC_CFLAGS=-ggdb -fopenmp -I../../include -I${MPICH_DIR}/include --offload-arch=gfx90a 

# Linker flags for hipcc on AMD platform
HIPCC_LDFLAGS=-L$(MPICH_DIR)/lib -lmpi $(PE_MPICH_GTL_DIR_amd_gfx90a) $(PE_MPICH_GTL_LIBS_amd_gfx90a) $(BASE_LIB_FLAGS)

# Compile the kernel source only with hipcc
%.exe : %.cpp
	hipcc $(HIPCC_CFLAGS) $< -o $@ $(HIPCC_LDFLAGS)

# Clean step
clean:
	rm -rf *.exe

.EXPORT_ALL_VARIABLES:
