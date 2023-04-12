# The HIP framework for parallel computing

The Heterogeneous Interface for Portability (HIP) provides a programming framework for harnessing the compute capabilities of multicore processors such as the AMD MI250X GPU’s on Setonix. In this course we focus on the essentials of developing HIP applications, with a focus on supercomputing.

## Folder structure

* **course_material** - contains the material for the course
* **deployment** - contains tools for deploying the material to Github and managing users
* **resources** - helpful tools and information for use wi the course

## Syllabus

In this course we cover the following topics. Each topic is a folder in **course_material**

* Lesson 1 - Introduction to HIP and high level features
* Lesson 2 - How to build and run applications on Cray AMD systems like Frontier and Setonix
* Lesson 3 - A complete example of matrix multiplication, explained line by line
* Lesson 4 - Debugging HIP applications
* Lesson 5 - Measuring the performance of HIP applications with profiling and tracing tools
* Lesson 6 - Memory management
* Lesson 7 - Strategies for optimising HIP kernels
* Lesson 8 - Strategies for optimising IO in HIP applications
* Lesson 9 - Porting CUDA applications to HIP

## Format

Each lesson is in the form of Jupyter notebooks which can be viewed on the student's machine with JupyterLab or with a web browser. All exercises may be performed on the command line using an SSH connection to a remote server.

## Installation

In order to use the material in this course a full installation of ROCM is advised, on distribution of Linux that ROCM supports and and AMD graphics card that ROCM supports. Alternatively the course may be run on a CUDA backened with a recent NVIDIA graphics card. In such instances the environment variable **HIP_PLATFORM** must be set to nvidia.


