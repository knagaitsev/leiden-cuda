#pragma once

#include <vector>
#include <cstdint>

void leiden(std::vector<uint32_t> offsets, std::vector<uint32_t> indices, std::vector<float> weights, float gamma);
