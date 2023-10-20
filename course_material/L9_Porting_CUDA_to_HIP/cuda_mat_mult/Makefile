# Applications

CXX=nvcc
# The -x cu flag is to tell nvcc to treat cpp files as cu files
CXXFLAGS=-g -O2 -Xcompiler -fopenmp -x cu
LIBFLAGS=-lcuda -lgomp

ifeq ($(CXX), hipcc)
    # If not using the NVIDIA backend
    ifneq ($(HIP_PLATFORM),nvidia)
        # Flags for AMD platform
        CXXFLAGS=-g -O2 -fopenmp
        LIBFLAGS=

        # Special flags for Setonix
        ifeq ($(findstring setonix,$(SLURM_SUBMIT_HOST)),setonix)
            CXXFLAGS="--offload-arch=gfx90a $(CXXFLAGS)"
        endif
    endif
endif

# List of applications to target
TARGETS=mat_mult.exe

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBFLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES: