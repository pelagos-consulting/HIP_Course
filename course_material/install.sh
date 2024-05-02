#!/bin/bash

# Get the location of this script
script_path=$(dirname "$0")

source $script_path/env

# Make the directory to build in
mkdir -p $script_path/build

#rm -rf $script_path/build/*

cd $script_path/build


# Run cmake
cmake -DCMAKE_INSTALL_MESSAGE=LAZY -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/ -DCMAKE_BUILD_TYPE=$BUILD_TYPE $COURSE_DIR/

#make clean
make install

