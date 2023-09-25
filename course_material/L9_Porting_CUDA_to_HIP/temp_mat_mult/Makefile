
# Applications

CXX=nvcc
CXXFLAGS=-g -O2 
LIBFLAGS=-lcuda

# List of applications to target
TARGETS=mat_mult.exe

all: $(TARGETS)

# General compilation step
%.exe: %.cu
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBFLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
