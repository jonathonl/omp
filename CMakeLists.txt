cmake_minimum_required(VERSION 3.5)
project(omp)

set(CMAKE_CXX_STANDARD 11)

find_package(Threads)

include_directories(include)

add_library(omp src/omp.cpp include/omp.hpp)

add_executable(omp-test main.cpp)
target_link_libraries(omp-test omp ${CMAKE_THREAD_LIBS_INIT})

install(FILES include/omp.hpp DESTINATION include)
install(TARGETS omp
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
