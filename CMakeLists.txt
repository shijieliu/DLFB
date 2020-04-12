cmake_minimum_required(VERSION 3.5)
project(LightLR)

set(CMAKE_CXX_STANDARD 11)

include_directories(./include)
#add_executable(test_tensor test/test_tensor.cpp src/tensor.cpp src/utils/random.cpp)
add_executable(test_node test/test_node.cpp src/node.cpp src/graph.cpp src/operator/add.cpp src/tensor.cpp)