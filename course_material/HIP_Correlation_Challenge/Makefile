
# Include general environment variables
include ../env

# Location of general helper files
INC_DIR=../include

# List of applications to target
TARGETS=xcorr_answers.exe xcorr.exe

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -I$(OPENCL_INC_DIR) -I$(INC_DIR) -L$(OPENCL_LIB_DIR) $<\
		-o $@ $(OPENCL_LIB_FLAGS) -lclblast

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
