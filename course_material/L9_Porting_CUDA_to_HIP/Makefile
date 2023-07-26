
# Include general environment variables
include ../env

# List of applications to target
TARGETS=wave2d_sync.exe wave2d_async.exe

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -r *.exe

.EXPORT_ALL_VARIABLES:
