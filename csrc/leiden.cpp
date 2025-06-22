#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

extern "C" void launch_add_kernel(float* a, float* b, float* c, int N);

extern "C" void move_nodes_fast(uint32_t *offsets, uint32_t *indices, float *weights, uint32_t *communities, int vertex_count, int edge_count);

void leiden(std::vector<uint32_t> offsets, std::vector<uint32_t> indices, std::vector<float> weights) {
    int vertex_count = offsets.size() - 1;
    int edge_count = indices.size();

    std::vector<uint32_t> communities;
    for (uint32_t i = 0; i < vertex_count; i++) {
        communities.push_back(i);
    }

    float gamma = 0.05;

    move_nodes_fast(offsets.data(), indices.data(), weights.data(), communities.data(), vertex_count, edge_count);

    for (int i = 0; i < vertex_count; i++) {
        std::cout << "Community " << i << ": " << communities[i] << "\n";
    }
}
