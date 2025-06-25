cmake_minimum_required(VERSION 3.18)
project(myleiden LANGUAGES CXX CUDA)

# find_package(pybind11 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# add_library(myleiden SHARED
#     src/leiden_kernel.cu
#     src/leiden.cpp
#     src/bindings.cpp
# )

# target_include_directories(myleiden PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )

# target_link_libraries(myleiden PRIVATE pybind11::module)

# set_target_properties(myleiden PROPERTIES
#     PREFIX ""
#     CUDA_SEPARABLE_COMPILATION ON
# )

add_executable(leiden_test
    src/leiden_kernel.cu
    src/leiden.cpp
    test/stopwatch_linux.cpp
    test/main.cpp
)

target_include_directories(leiden_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(leiden_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
