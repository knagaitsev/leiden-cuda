cmake_minimum_required(VERSION 3.18)
project(cugraph_leiden_simple_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find all dependencies
find_package(CUDAToolkit REQUIRED)
find_package(RMM REQUIRED)
find_package(RAFT REQUIRED)
find_package(cuGraph REQUIRED)

# Define the executable
add_executable(leiden_cugraph leiden_cugraph.cpp)

# Link against all required libraries
target_link_libraries(leiden_cugraph
    PRIVATE
        rmm::rmm
        raft::raft
        cugraph::cugraph
)
