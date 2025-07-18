#include "leiden/leiden_kernel.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

void leiden(std::vector<uint32_t> offsets, std::vector<uint32_t> indices, std::vector<float> weights, std::vector<uint32_t> full_edge_list_u, std::vector<uint32_t> full_edge_list_v, float gamma) {
    int vertex_count = offsets.size() - 1;
    int edge_count = indices.size();

    std::vector<node_data_t> node_data;

    std::vector<uint32_t> node_agg_counts;

    std::vector<comm_data_t> comm_data;

    for (uint32_t i = 0; i < vertex_count; i++) {
        node_data_t node = { .community = i };
        node_data.push_back(node);

        node_agg_counts.push_back(1);

        comm_data_t comm = { .agg_count = 1 };
        comm_data.push_back(comm);

        // uint32_t offset = offsets[i];
        // uint32_t offset_next = offsets[i + 1];
        // uint32_t node_edge_count = offset_next - offset;
        // std::cout << "Node " << i << " edge count: " << node_edge_count << "\n";
    }

    int comm_count = vertex_count;

    leiden_internal(offsets.data(), indices.data(), weights.data(), full_edge_list_u.data(), full_edge_list_v.data(), node_data.data(), node_agg_counts.data(), comm_data.data(), vertex_count, edge_count, comm_count, gamma);

    // for (int i = 0; i < vertex_count; i++) {
    //     std::cout << "Vertex " << i << " community: " << node_data[i].community << "\n";
    // }
}
