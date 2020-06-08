cmake_minimum_required(VERSION 3.5)
project(DL)

set(CMAKE_CXX_STANDARD 11)

include_directories(./include)
add_executable(test_tensor example/test_tensor.cpp)
add_executable(test_graph example/test_graph.cpp src/graph.cpp src/node.cpp)