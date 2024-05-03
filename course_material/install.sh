#!/bin/bash

# Get the location of this script
script_path=$(dirname "$0")

env_script=$(realpath $script_path/env) 

source $env_script

# Make the directory to build in
mkdir -p $COURSE_DIR/build

cd $COURSE_DIR/build

# Run cmake
cmake -DCMAKE_INSTALL_MESSAGE=LAZY -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/ -DCMAKE_BUILD_TYPE=$BUILD_TYPE $COURSE_DIR/

#make clean
make all install

build_script=${RUN_DIR}/build

# Make a build script
cat << EOF > ${build_script}
#!/bin/bash

source ${env_script}





EOF

# Change permissions
chmod u+x ${build_script} 
