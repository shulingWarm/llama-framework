#cmake的最小版本
cmake_minimum_required(VERSION 3.2)
project(llamaFramework)
include_directories(${CMAKE_SOURCE_DIR})
find_package(CUDA REQUIRED)
cuda_add_executable(directDOM directDOM.cu)