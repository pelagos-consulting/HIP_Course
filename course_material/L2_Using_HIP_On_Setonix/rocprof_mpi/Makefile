
# List of applications to target
TARGETS=mat_mult_profiling_mpi.exe

all: $(TARGETS)

# Compiler flags for hipcc on AMD platform
HIPCC_CFLAGS=-ggdb -fopenmp -I../../common -I${MPICH_DIR}/include --offload-arch=gfx90a 

# Linker flags for hipcc on AMD platform
HIPCC_LDFLAGS=-L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa

# Compile the kernel source only with hipcc
%.exe : %.cpp
	hipcc $(HIPCC_CFLAGS) ../../common/hip_helper.cpp $< -o $@ $(HIPCC_LDFLAGS)

# Clean step
clean:
	rm -rf *.exe

.EXPORT_ALL_VARIABLES:
