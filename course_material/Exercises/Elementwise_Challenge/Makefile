
# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_elementwise.exe mat_elementwise_answer.exe

all: $(TARGETS)

# Kernel dependencies

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
