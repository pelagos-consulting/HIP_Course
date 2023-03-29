
# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_mult_coarse_buffer_svm.exe mat_mult_fine_buffer_svm.exe svm_check.exe mat_mult_fine_system_svm.exe mat_elementwise_svm.exe mat_elementwise_svm_answer.exe

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
