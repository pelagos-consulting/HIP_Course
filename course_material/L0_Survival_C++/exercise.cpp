#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
    // Print all the arguments passed into the application
    for (int i=0; i<argc; i++) {
        std::printf("%s ", argv[i]);
    }
    
    // Put a new line 
    std::printf("\n");

    std::cout << "Hello" << "\n";
}
