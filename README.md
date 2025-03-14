# Accelerated computing with HIP

The Heterogeneous Interface for Portability (HIP) provides a programming framework for harnessing the compute capabilities of multicore processors such as the AMD MI250X GPU’s on Setonix. In this course we focus on the essentials of developing HIP applications, with a focus on supercomputing.

## Folder structure

* **course_material** - contains the course material.
* **deployment** - contains tools for deploying course material to Github and managing users.
* **resources** - helpful tools and information for use within the course.

## Syllabus

In this course we cover the following topics. Each topic is a subfolder in **course_material**.

* Lesson 1 - Introduction to HIP and high level features
* Lesson 2 - How to build and run applications on Cray AMD systems like Frontier and Setonix
* Lesson 3 - A complete example of matrix multiplication, explained line by line
* Lesson 4 - Debugging HIP applications
* Lesson 5 - Measuring the performance of HIP applications with profiling and tracing tools
* Lesson 6 - Memory management with HIP
* Lesson 7 - Strategies for improving the performance of HIP kernels
* Lesson 8 - Strategies for optimising application performance with concurrent IO.
* Lesson 9 - Porting CUDA applications to HIP

## Format

Lessons are in the form of Jupyter notebooks which can be viewed on the student's machine with JupyterLab or with a web browser. All exercises may be performed on the command line using an SSH connection to a remote server that has ROCM installed.

## Installation

The course material uses a CMake build system. You can use either an AMD or NVIDIA backend and will need the following software installed and available:

* A Linux environment with graphics card drivers installed
* CMake - version 3.21+
* ROCM - with HipBLAS (this has to be installed regardless of which backend you use)
* Text editor

In the file [course_material/env](course_material/env) is a BASH script that loads modules, sets environment variables, and sets install and run paths. You **will need** to customise this file for your needs. The main variables that need to be edited are the `HIP_PLATFORM` and `GPU_ARCH` variables. 

Once this is done you can run the [course_material/install.sh](course_material/install.sh) script to install the software into $INSTALL_DIR. When running examples you can source `course_material/env` to set load modules and set paths.

### Linux environment

It is the path of least pain to use a distribution of Linux that ROCM officially supports. Then you can use a package manager to install all the dependencies.

### ROCM

Regardless of which backend you plan to use, the ROCM framework must be available, and a full installation of [ROCM](https://docs.amd.com/) is advised in order to get all the necessary tools like debuggers and profilers. In particular this course needs the following ROCM packages installed and available.

* hip
* hip-dev
* hipblas
* hipblas-dev
* rocprofiler
* rocprofiler-dev
* rocm-gdb
* rocm-openmp-sdk

### AMD backend

It is advisable to use an AMD graphics card that ROCM supports. Other AMD GPU's work unofficially, but you can expect the occasional undefined behaviour and things not working like they should. I have had good success with both supported and non-supported graphics cards when I **avoided installing the amdgpu-dkms** package and used the AMD graphics card drivers that are built into the Linux kernel.

Within `course_material/env` you must set `HIP_PLATFORM` to `amd`. Then you need to put into `GPU_ARCH` a semicolon list of GPU architectures that you'd like to build for. For example, on an AMD MI250X the GPU architecture is `gfx90a` and on an AMD MI300 the GPU architecture is `gfx942`. 

### NVIDIA backend

HIP can also use a CUDA backend. In that case install both ROCM and a compatible version of [CUDA](https://developer.nvidia.com/cuda-downloads). Usually a safe choice for CUDA is the prior major release. Then the course may be run on a CUDA backend with a recent NVIDIA graphics card. In such instances set the environment variable `HIP_PLATFORM` to `nvidia` in `course_material/dev`.

```bash
export HIP_PLATFORM=nvidia
```

Then within `course_material/dev` set `GPU_ARCH` to a semicolon separated list of GPU architectures you would like to have available. For my NVIDIA RTX 3060 this is just `86`, and on an NVIDIA Tesla T4 it is `75`. You can support more than one architecture.

### Anaconda Python (optional)

A recent version of [Anaconda Python](https://www.anaconda.com/products/distribution) is helpful for viewing the notebook files. Once Anaconda is installed create a new environment 

```bash
conda create --name hip_course python=3.10 nodejs=18.15.0
conda activate hip_course
```
A list of helpful packages for viewing the material may then be installed with this command when run from the **HIP_Course** folder. 

```bash
pip install -r ./deployment/python_packages.txt
```
then run 

```bash
jupyter-lab
```
from the command line to start the Jupyter Lab environment. The command

```bash
conda activate hip_course
```
is to enter the created environment, and the command
```bash
conda deactivate
```
will leave the environment.


