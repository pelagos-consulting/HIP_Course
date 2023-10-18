# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_mult_float.exe \
		mat_mult_float_restrict.exe \
		mat_mult_double.exe \
		mat_mult_AT.exe \
		mat_mult_BT.exe \
		mat_mult_shared_A.exe \
		mat_mult_shared_B.exe \
		mat_mult_tile_shared_A.exe \
		mat_mult_tile_shared_B.exe \
		mat_mult_tile_shared_AB.exe \
		mat_mult_tile_shared_A_vector.exe \
		mat_mult_tile_shared_B_vector.exe \
		mat_mult_tile_shared_AB_vector.exe \
		mat_mult_float_hipblas.exe \
		mat_mult_float_md_hipblas.exe

# Select the right platform
ifeq ($(HIP_PLATFORM), nvidia)
	# pass
	ANALYSIS_FLAGS=
else
    # Normal compilation
    ANALYSIS_FLAGS=-Rpass-analysis=kernel-resource-usage
endif

all: $(TARGETS)

# General compilation step
%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(ANALYSIS_FLAGS) $(BASE_INC_FLAGS) $< -o $@ 

# General linking step
%.exe: %.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Special compilation for hipblas codes
%hipblas.o: %hipblas.cpp
	$(CXX) -c $(CXXFLAGS) $(ANALYSIS_FLAGS) $(BASE_INC_FLAGS) $< -o $@ 

%hipblas.exe: %hipblas.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(BASE_LIB_FLAGS) -lhipblas

# Clean step
clean:
	rm -rf *.exe


.EXPORT_ALL_VARIABLES:


