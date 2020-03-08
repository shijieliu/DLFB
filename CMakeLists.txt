cmake_minimum_required(VERSION 3.5)
project(LightLR)

set(CMAKE_CXX_STANDARD 11)

include_directories(./include)
#add_executable(LightLR main.cpp)
add_executable(test_tensor test/test_tensor.cpp src/tensor.cpp)