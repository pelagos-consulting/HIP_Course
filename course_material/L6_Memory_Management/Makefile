# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_mult_pinned_mem.exe \
		mat_mult_managed_mem.exe \
		mat_mult_pitched_mem.exe \
		mat_mult_shared_A.exe \
		mat_mult_shared_A_faulty.exe \
		mat_mult_shared_A_vector.exe \
		vector_types.exe \
		atomics.exe \
		mat_elementwise.exe \
		mat_elementwise_answer.exe        

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -rf *.exe


.EXPORT_ALL_VARIABLES:
