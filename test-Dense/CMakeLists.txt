cmake_minimum_required(VERSION 3.8)
project(Code)

set(CMAKE_CXX_STANDARD 11)
set(CUSNN_INC /home/fede/Software/cuSNN/include)
set(CUSNN_LIB /home/fede/Software/cuSNN/lib/libcuSNN.so)

set(SOURCE_FILES
        build/main
        src/data.cpp
        src/data.h
        src/main.cpp
        src/plotter.cpp
        src/plotter.h
        ${CUSNN_INC}/cusnn.cuh
        ${CUSNN_INC}/cusnn_kernels.cuh
        Makefile)

add_executable(Code ${SOURCE_FILES})
target_include_directories(Code PUBLIC ${CUSNN_INC})
target_link_libraries(Code ${CUSNN_LIB})