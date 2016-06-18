cmake_minimum_required(VERSION 3.5)
project(omp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(include)

add_library(omp src/omp.cpp include/omp.hpp)

add_executable(omp-test main.cpp)
target_link_libraries(omp-test omp)