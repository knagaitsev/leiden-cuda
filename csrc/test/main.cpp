#include "leiden/leiden.hpp"
#include "leiden/test/stopwatch_linux.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <stdexcept>

typedef std::vector<std::pair<std::pair<uint32_t, uint32_t>, float>> edge_list_t;

typedef std::pair<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, std::vector<float>>> offsets_indices_weights_t;

class EdgeStore {
public:
    // Vector of pairs: (vertex_id, weight)
    std::vector<std::pair<uint32_t, float>> edge_weights;

    // Set of unique vertex IDs
    std::set<uint32_t> vertices;

    // Insert method: only insert if vertex not already present
    bool insert(std::pair<uint32_t, float> vertex_id_weight) {
        if (vertices.find(vertex_id_weight.first) != vertices.end()) {
            return false; // Already exists
        }
        edge_weights.push_back(vertex_id_weight);
        vertices.insert(vertex_id_weight.first);
        return true;
    }
};

struct csr_result_t {
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> indices;
    std::vector<float> weights;
    std::vector<uint32_t> full_edge_list_u;
    std::vector<uint32_t> full_edge_list_v;
};

// TODO: make self-edges and duplicate edges work here
csr_result_t to_csr(edge_list_t edge_list) {
    std::map<uint32_t, EdgeStore> vertex_map;
    int edge_count = 0;

    for (auto edge_data : edge_list) {
        auto edge = edge_data.first;
        auto weight = edge_data.second;
        auto u = edge.first;
        auto v = edge.second;

        auto new_edge_to_u = std::make_pair(u, weight);
        auto new_edge_to_v = std::make_pair(v, weight);

        if (vertex_map.find(u) != vertex_map.end()) {
            edge_count += vertex_map[u].insert(new_edge_to_v);
        } else {
            EdgeStore neighbors;
            neighbors.insert(new_edge_to_v);
            vertex_map[u] = neighbors;
            edge_count++;
        }

        if (vertex_map.find(v) != vertex_map.end()) {
            vertex_map[v].insert(new_edge_to_u);
        } else {
            EdgeStore neighbors;
            neighbors.insert(new_edge_to_u);
            vertex_map[v] = neighbors;
        }
    }

    std::cout << "Edge count: " << edge_count << "\n";

    csr_result_t result;

    auto curr_offset = 0;
    auto idx = 0;
    auto nodes_without_neighbors_count = 0;

    for (auto kv : vertex_map) {
        while (idx < kv.first) {
            nodes_without_neighbors_count++;
            result.offsets.push_back(curr_offset);
            idx++;
        }

        auto edge_weights = kv.second.edge_weights;
        std::sort(edge_weights.begin(), edge_weights.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        for (auto edge_data : edge_weights) {
            result.full_edge_list_u.push_back(kv.first);
            result.full_edge_list_v.push_back(edge_data.first);
            result.indices.push_back(edge_data.first);
            result.weights.push_back(edge_data.second);
        }

        result.offsets.push_back(curr_offset);
        curr_offset += edge_weights.size();
        idx++;
    }

    result.offsets.push_back(curr_offset);

    if (nodes_without_neighbors_count > 0) {
        std::cout << "WARNING: got " << nodes_without_neighbors_count << " nodes without neighbors\n";
    }

    return result;
}

edge_list_t load_edge_list(std::string filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw new std::runtime_error("Failed to open file");
    }

    edge_list_t edge_list;

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty/whitespace or comment line
        if (line.empty() || line[0] == '%') {
            continue;
        }

        std::istringstream iss(line);
        uint32_t a, b;
        if (iss >> a >> b) {
            auto weight = 1.0;

            // if (a == b) {
            //     std::cout << "Dataset contains self-edge, removing this edge as they are currently unsupported\n";
            //     continue;
            // }

            edge_list.push_back(std::make_pair(std::make_pair(a, b), weight));
        }
    }

    file.close();

    if (edge_list.size() == 0) {
        throw std::runtime_error("edge_list empty");
    }

    auto min_vertex_idx = edge_list[0].first.first;

    for (auto &edge_data : edge_list) {
        auto edge = edge_data.first;
        auto u = edge.first;
        auto v = edge.second;

        if (u < min_vertex_idx) {
            min_vertex_idx = u;
        }

        if (v < min_vertex_idx) {
            min_vertex_idx = v;
        }
    }

    std::cout << "Min vertex idx: " << min_vertex_idx << "\n";

    for (auto &edge_data : edge_list) {
        edge_data.first.first -= min_vertex_idx;
        edge_data.first.second -= min_vertex_idx;
    }

    return edge_list;
}

int main() {
    // auto filename = std::string("validation/clique_ring.txt");
    auto filename = std::string("data/wikipedia_link_mi/out.wikipedia_link_mi");
    // auto filename = std::string("data/arenas-jazz/out.arenas-jazz");
    // auto filename = std::string("data/flickr-groupmemberships/out.flickr-groupmemberships");
    // auto filename = std::string("data/youtube-links/out.youtube-links");
    // auto filename = std::string("data/flickr-links/out.flickr-links");

    auto edge_list = load_edge_list(filename);
    auto csr_result = to_csr(edge_list);
    auto offsets = csr_result.offsets;
    auto indices = csr_result.indices;
    auto weights = csr_result.weights;
    auto full_edge_list_u = csr_result.full_edge_list_u;
    auto full_edge_list_v = csr_result.full_edge_list_v;

    std::cout << "Vertex count: " << (offsets.size() - 1) << "\n";
    std::cout << "Offsets size: " << offsets.size() << "\n";
    std::cout << "Indices size: " << indices.size() << "\n";
    std::cout << "Weights size: " << weights.size() << "\n";

    std::cout << "full_edge_list_u size: " << full_edge_list_u.size() << "\n";
    std::cout << "full_edge_list_v size: " << full_edge_list_v.size() << "\n";

    float gamma = 0.05;

    auto stopwatch = StopWatchLinux();
    stopwatch.start();
    leiden(offsets, indices, weights, full_edge_list_u, full_edge_list_v, gamma);
    stopwatch.stop();

    auto time_ms = stopwatch.getTime();
    auto time_s = time_ms / 1000;

    std::cout << "Runtime: " << time_s << "s\n";

    return 0;
}
