#include <vector>

extern "C" void launch_add_kernel(float* a, float* b, float* c, int N);

void leiden(float *a, float *b, float *c, int N) {
    launch_add_kernel(a, b, c, N);
}
