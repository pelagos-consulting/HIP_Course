
# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_mult_local.exe mat_mult_local_transpose_vector.exe atomics.exe atomics2.exe mat_elementwise.exe mat_elementwise_answer.exe

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
