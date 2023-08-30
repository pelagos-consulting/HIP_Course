# Include general environment variables
include ../env

# List of applications to target
TARGETS=mat_mult_float.exe \
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
		mat_mult_float_hipblas.exe        
#		vector_types.exe \
#		atomics.exe \
#		mat_elementwise.exe \
#		mat_elementwise_answer.exe        

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -rf *.exe


.EXPORT_ALL_VARIABLES:


