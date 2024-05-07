# Include general environment variables
include ../env_make

# List of applications to target
TARGETS=mat_mult.exe

all: $(TARGETS)

# Kernel dependencies

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
