#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

extern "C" void launch_add_kernel(float* a, float* b, float* c, int N);

int main() {
    std::ifstream file("validation/clique_ring.txt");

    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    std::vector<std::pair<uint32_t, uint32_t>> edge_list;

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty/whitespace or comment line
        if (line.empty() || line[0] == '%') {
            continue;
        }

        std::istringstream iss(line);
        uint32_t a, b;
        if (iss >> a >> b) {
            edge_list.push_back(std::make_pair(a, b));
        }
    }

    file.close();

    std::cout << "Edge list len: " << edge_list.size() << "\n";

    // int N = 1024;
    // float *a = new float[N];
    // float *b = new float[N];
    // float *c = new float[N];

    // for (int i = 0; i < N; ++i) {
    //     a[i] = 1.0f;
    //     b[i] = 2.0f;
    // }

    // launch_add_kernel(a, b, c, N);

    // std::cout << "Result: ";
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << c[i] << " ";
    // }
    // std::cout << "...\n";

    // delete[] a;
    // delete[] b;
    // delete[] c;
    return 0;
}
