# Set debugging flag
DEBUG=true

# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_mult.exe mat_mult_bug.exe mat_mult_memcheck.exe mat_mult_printf.exe mat_elementwise_answer.exe mat_elementwise.exe

all: $(TARGETS) 

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
