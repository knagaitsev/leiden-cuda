#include "leiden/leiden_kernel.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

void leiden(std::vector<uint32_t> offsets, std::vector<uint32_t> indices, std::vector<float> weights) {
    int vertex_count = offsets.size() - 1;
    int edge_count = indices.size();

    std::vector<node_data_t> node_data;

    std::vector<comm_data_t> comm_data;

    for (uint32_t i = 0; i < vertex_count; i++) {
        node_data_t node = { .community = i, .agg_count = 1 };
        node_data.push_back(node);

        comm_data_t comm = { .agg_count = 1 };
        comm_data.push_back(comm);

        // uint32_t offset = offsets[i];
        // uint32_t offset_next = offsets[i + 1];
        // uint32_t node_edge_count = offset_next - offset;
        // std::cout << "Node " << i << " edge count: " << node_edge_count << "\n";
    }

    int comm_count = vertex_count;

    float gamma = 0.05;

    move_nodes_fast(offsets.data(), indices.data(), weights.data(), node_data.data(), comm_data.data(), vertex_count, edge_count, comm_count, gamma);

    std::cout << "\n";

    // for (int i = 0; i < vertex_count; i++) {
    //     std::cout << "Vertex " << i << " community: " << node_data[i].community << "\n";
    // }
}
