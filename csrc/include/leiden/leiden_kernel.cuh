#pragma once

#include <cstdint>

typedef struct node_data {
    uint32_t community;
    uint32_t agg_count;
} node_data_t;

typedef struct comm_data {
    uint32_t agg_count;
} comm_data_t;

typedef struct part_scan_data {
    uint32_t scanned_agg_count;
    uint32_t curr_node_idx;
} part_scan_data_t;

extern "C" void move_nodes_fast(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    comm_data_t *comm_data,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma
);
