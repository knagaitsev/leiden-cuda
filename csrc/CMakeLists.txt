add_library(myleiden SHARED leiden_kernel.cu leiden.cpp bindings.cpp)

target_link_libraries(myleiden PRIVATE pybind11::module)

set_target_properties(myleiden PROPERTIES
    PREFIX ""  # Prevents lib prefix: gives myleiden.so instead of libmycuda.so
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(leiden_test leiden_kernel.cu leiden.cpp main.cpp)
set_target_properties(leiden_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(myleiden PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
