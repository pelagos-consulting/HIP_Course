

# Location of the OpenCL library, where libOpenCL.so resides
CL_LIB=/etc/alternatives/opencl-intel-tools/lib64
# Location of the OpenCL CL directory where cl.h and cl.hpp reside
CL_INCLUDE=/etc/alternatives/opencl-intel-tools/include

# C++ compiler and flags
CXX=g++

ifeq ($(OS),Windows_NT)
	CXXFLAGS=-g -O3 -fPIC -I$(CL_INCLUDE) -std=c++11
	LFLAGS=-g -L$(CL_LIB) -lOpenCL -lstdc++         
else
	uname_s := $(shell uname -s)
	ifeq ($(uname_s),Linux)
		CXXFLAGS=-g -O3 -fPIC -I$(CL_INCLUDE) -std=c++11
		LFLAGS=-g -L$(CL_LIB) -lstdc++ -lOpenCL         
	endif
	ifeq ($(uname_s),Darwin)
		CXXFLAGS=-g -O3 -fPIC -std=c++11
		LFLAGS=-g -lstdc++ -framework OpenCL        
	endif
endif

# Matrix multiplication
all:    mat_mult \
	copy_rect_region \
	mat_mult_create_binary \
	mat_mult_use_binary \
	mat_mult_transpose \
	mat_mult_transpose_vector \
    template

mat_mult:	mat_mult.o
	$(CXX) $(LFLAGS) -o $@ $<

mat_mult_create_binary:	mat_mult_create_binary.o
	$(CXX) $(LFLAGS) -o $@ $<

mat_mult_use_binary:	mat_mult_use_binary.o
	$(CXX) $(LFLAGS) -o $@ $<

copy_rect_region:	copy_rect_region.o
	$(CXX) $(LFLAGS) -o $@ $<

mat_mult_transpose:	mat_mult_transpose.o
	$(CXX) $(LFLAGS) -o $@ $<

mat_mult_transpose_vector:	mat_mult_transpose_vector.o
	$(CXX) $(LFLAGS) -o $@ $<

template:	template.o
	$(CXX) $(LFLAGS) -o $@ $<

%.o:	%.cpp helper_functions.hpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

clean:
	rm -rf *.o *.mod *.bin \
    mat_mult \
    copy_rect_region \
    mat_mult_create_binary \
    mat_mult_use_binary \
    mat_mult_transpose \
    mat_mult_transpose_vector \
    template
