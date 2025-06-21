add_library(mycuda SHARED my_kernel.cu bindings.cpp)

target_link_libraries(mycuda PRIVATE pybind11::module)

set_target_properties(mycuda PROPERTIES
    PREFIX ""  # Prevents lib prefix: gives mycuda.so instead of libmycuda.so
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(mycuda_test main.cpp my_kernel.cu)
set_target_properties(mycuda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(mycuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
