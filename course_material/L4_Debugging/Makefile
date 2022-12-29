
# Include general environment variables
include ../env

# Location of general helper files
INC_DIR=../include

# List of applications to target
TARGETS=mat_mult.exe mat_mult_memcheck.exe mat_mult_badmem.exe mat_mult_printf.exe mat_mult_badmem_gdb.exe mat_elementwise_buggy.exe mat_elementwise_answer.exe

all: $(TARGETS) 

# Override CXXFLAGS
CXXFLAGS := $(CXXFLAGS) -O0

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -I$(OPENCL_INC_DIR) -I$(INC_DIR) -L$(OPENCL_LIB_DIR) $<\
		-o $@ $(OPENCL_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
