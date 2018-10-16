cmake_minimum_required(VERSION 3.8)
project(example_train)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        build/main
        obj/data.o
        obj/plotter.o
        obj/main.o
        src/data.cpp
        src/data.h
        src/main.cpp
        src/plotter.cpp
        src/plotter.h
        Makefile)

add_executable(example_train ${SOURCE_FILES})