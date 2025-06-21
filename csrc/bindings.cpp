#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" void launch_add_kernel(float* a, float* b, float* c, int N);

namespace py = pybind11;

py::array_t<float> py_add(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request(), buf_b = b.request();
    if (buf_a.size != buf_b.size)
        throw std::runtime_error("Input sizes must match");

    py::array_t<float> result(buf_a.size);
    auto buf_c = result.request();

    launch_add_kernel((float*)buf_a.ptr, (float*)buf_b.ptr, (float*)buf_c.ptr, buf_a.size);

    return result;
}

PYBIND11_MODULE(mycuda, m) {
    m.def("add", &py_add, "Add two arrays on GPU");
}
