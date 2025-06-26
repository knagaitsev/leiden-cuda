#pragma once

#include <cstdint>

typedef struct node_data {
    uint32_t community;
} node_data_t;

typedef struct comm_data {
    uint32_t agg_count;
} comm_data_t;

typedef struct part_scan_data {
    uint32_t scanned_agg_count;
    uint32_t curr_node_idx;
} part_scan_data_t;

typedef struct weight_idx {
    float weight;
    uint32_t idx;
} weight_idx_t;

void leiden_internal(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    uint32_t *full_edge_list_u,
    uint32_t *full_edge_list_v,
    node_data_t *node_data,
    uint32_t *node_agg_counts,
    comm_data_t *comm_data,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma
);
