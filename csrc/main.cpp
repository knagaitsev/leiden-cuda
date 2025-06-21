#include <iostream>

extern "C" void launch_add_kernel(float* a, float* b, float* c, int N);

int main() {
    int N = 1024;
    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    launch_add_kernel(a, b, c, N);

    std::cout << "Result: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << "...\n";

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}
