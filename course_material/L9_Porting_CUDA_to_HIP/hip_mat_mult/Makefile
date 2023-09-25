
# Applications

CXX=hipcc
CXXFLAGS=-g -O2
# Library flags
LIBFLAGS=

# Special flags for Setonix
ifeq ($(findstring setonix,$(SLURM_SUBMIT_HOST)),setonix)
    CXXFLAGS:=--offload-arch=gfx90a $(CXXFLAGS)
endif

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
