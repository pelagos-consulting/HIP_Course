
# Applications

CXX=nvcc
CXXFLAGS=-g -O2 

# List of applications to target
TARGETS=mat_mult.exe

all: $(TARGETS)

# Kernel dependencies

# General compilation step
%.exe: %.cu
	$(CXX) $(CXXFLAGS) $< -o $@ 

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
