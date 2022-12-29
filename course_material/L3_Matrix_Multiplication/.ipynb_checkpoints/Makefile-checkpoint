
# Include general environment variables
include ../env

# Location of general helper files
INC_DIR=../include

# List of applications to target
TARGETS=mat_mult.exe mat_elementwise.exe mat_elementwise_answer.exe

all: $(TARGETS)

# Kernel dependencies
mat_elementwise.exe: kernels_elementwise.c
mat_elementwise_answer.exe: kernels_elementwise_answer.c
mat_mult.exe: kernels_mat_mult.c

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -I$(OPENCL_INC_DIR) -I$(INC_DIR) -L$(OPENCL_LIB_DIR) $<\
		-o $@ $(OPENCL_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
