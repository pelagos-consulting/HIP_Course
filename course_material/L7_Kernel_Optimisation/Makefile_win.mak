# Run this file as

# nmake /f Makefile_win

# Include general environment variables
!include ..\env.mak

# Location of general helper files
INC_DIR=..\include

# List of applications to target
TARGETS=mat_mult_float.exe mat_mult_double.exe mat_mult_local.exe mat_mult_prefetch.exe mat_mult_transpose_A.exe mat_mult_transpose_B.exe mat_mult_tile_local.exe mat_mult_tile_local_vector.exe mat_mult_chunk_vector.exe mat_mult_clblast.exe mat_mult_clblast_md.exe

all: $(TARGETS)

CLBLAST_LIB_DIR="C:\Program Files\CLBlast\1.5.2\lib"
CLBLAST_INC_DIR="C:\Program Files\CLBlast\1.5.2\include"

# General compilation step
.cpp.exe:
	$(CXX) $(CXXFLAGS) /I $(INC_DIR) /I $(CLBLAST_INC_DIR) $< $(OPENCL_LIB_FLAGS) $(CLBLAST_LIB_DIR)/clblast.lib -o $@  

# Clean step
clean:
	DEL /Q /F /S "*.obj"
	DEL /Q /F /S "*.exe"


.EXPORT_ALL_VARIABLES:
