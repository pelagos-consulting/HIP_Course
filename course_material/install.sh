#!/bin/bash

# Get the location of this script
script_path=$(dirname "$0")

# This script will load any modules and get the environment ready
env_script=$(realpath $script_path/env) 
source $env_script

# Make the directory to build in
build_dir=$COURSE_DIR/build
mkdir -p $build_dir
cd $build_dir

# Run cmake
cmake -DCMAKE_INSTALL_MESSAGE=NEVER -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_BUILD_TYPE=$BUILD_TYPE $COURSE_DIR/

# Make everything and install
make all install

# Write a build script
build_script=${RUN_DIR}/build

cat << EOF > ${build_script}
#!/bin/bash --login
source ${env_script}
cd $build_dir
if [ "\$#" -eq 1 ]
then
    make \$1 install    
else
    make all install
fi
EOF

# Change permissions
chmod u+x ${build_script} 

# Write a run script
run_script=${RUN_DIR}/run

# Make a run script
cat << EOF > ${run_script}
#!/bin/bash --login
source ${env_script}
# Run all arguments passed to the script
\$@
EOF

# Change permissions
chmod u+x ${run_script}
