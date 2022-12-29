
# Include general environment variables
include ../env

# Location of general helper files
INC_DIR=../include

# List of applications to target
TARGETS=mat_mult_float.exe mat_mult_double.exe mat_mult_local.exe mat_mult_prefetch.exe mat_mult_transpose_A.exe mat_mult_transpose_B.exe mat_mult_tile_local.exe mat_mult_tile_local_vector.exe mat_mult_clblast.exe mat_mult_clblast_md.exe mat_mult_chunk_vector.exe

all: $(TARGETS)

CXXFLAGS=-g -O2 -fopenmp

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -I$(OPENCL_INC_DIR) -I$(INC_DIR) -L$(OPENCL_LIB_DIR) $<\
		-o $@ $(OPENCL_LIB_FLAGS) -lclblast

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
