
# Include general environment variables
include ../env

# Location of general helper files
INC_DIR=../include

# List of applications to target
TARGETS=mat_mult.exe

all: $(TARGETS)

# Kernel dependencies

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -I$(BASE_INC_DIR) -I$(INC_DIR) -L$(BASE_LIB_DIR) $<\
		-o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe


.EXPORT_ALL_VARIABLES:
