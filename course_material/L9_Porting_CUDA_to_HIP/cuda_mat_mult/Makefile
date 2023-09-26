
# Applications

CXX=nvcc
# The -x cu flag is to tell nvcc to treat cpp files as cu files
CXXFLAGS=-g -O2 -x cu
LIBFLAGS=-lcuda

# List of applications to target
TARGETS=mat_mult.exe

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBFLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
