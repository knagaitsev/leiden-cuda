#pragma once

#include <vector>
#include <cstdint>

void leiden(std::vector<uint32_t> offsets, std::vector<uint32_t> indices, std::vector<float> weights, std::vector<uint32_t> full_edge_list_u, std::vector<uint32_t> full_edge_list_v, float gamma);
