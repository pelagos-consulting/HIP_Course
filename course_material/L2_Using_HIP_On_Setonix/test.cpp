#include <cstdio>

int main(int argc, char** argv) {
	float a=1.0;

#ifdef __HIP_ROCclr__
	std::printf("Roclr is defined\n");
#endif

#ifdef __HIP_ARCH_GFX90A__
	std::printf("GFX90A is defined\n");
#endif

}
