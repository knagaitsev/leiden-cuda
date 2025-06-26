#include "leiden/leiden_kernel.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
// #include <cub/cub.cuh>

__global__ void init_rng(curandState *state, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void generate_random(float *random_numbers, curandState *state, int vertex_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= vertex_count) {
        return;
    }

    curandState local_state = state[tid];
    random_numbers[tid] = curand_uniform(&local_state); // Range: (0.0, 1.0]
    state[tid] = local_state; // Save state back
}

// two approaches to doing move_nodes_fast: parallelizing at node level is below
// - another option is parallelizing at edge level, letting each thread consider an edge
__global__ void move_nodes_fast_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    uint32_t *node_agg_counts,
    comm_data_t *comm_data,
    int *comm_locks,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma,
    bool *changed,
    uint32_t *partition,
    uint32_t *node_moves,
    uint32_t *node_to_comm_counts_final,
    uint32_t *node_to_comm_comms_final,
    float *node_to_comm_weights_final,
    float *d_random,
    bool *node_visited
) {
    unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= vertex_count) {
        return;
    }

    if (node_visited[node]) {
        return;
    }

    float rand = d_random[node];

    // printf("Node: %d, Rand: %f\n", node, rand);

    uint32_t offset = offsets[node];
    uint32_t offset_next = offset + node_to_comm_counts_final[node];
    // uint32_t offset_next = offsets[node + 1];

    for (uint32_t i = offset; i < offset_next; i++) {
        uint32_t neighbor = indices[i];
        if (neighbor == node) {
            continue;
        }
        // if your neighbor hasn't been visited and you have a smaller rand, the
        // neighbor wins in the graph coloring
        if (!node_visited[neighbor] && rand < d_random[neighbor]) {
            return;
        }
    }

    uint32_t curr_comm = node_data[node].community;

    uint32_t best_comm = curr_comm;
    float best_delta = 0.0f;

    // uint32_t node_edge_count = offset_next - offset;

    // aggregate count of nodes in old community (including current node)
    int agg_count_old = comm_data[curr_comm].agg_count;

    // aggregate count of current node
    int node_agg_count = node_agg_counts[node];

    // total edge weight of incoming edges from old community
    float k_vc_old = 0.0;
    // for (uint32_t i = offset; i < offset_next; i++) {
    //     uint32_t neigh = indices[i];
    //     if (node_data[neigh].community == curr_comm) {
    //         k_vc_old += weights[i];
    //     }
    // }

    for (uint32_t i = offset; i < offset_next; i++) {
        uint32_t neighbor = indices[i];
        uint32_t neighbor_comm = node_data[neighbor].community;
        if (neighbor_comm == curr_comm) {
            k_vc_old = node_to_comm_weights_final[i];
            break;
        }
    }

    for (uint32_t i = offset; i < offset_next; i++) {
        uint32_t neighbor = indices[i];
        uint32_t neighbor_comm = node_data[neighbor].community;
        // uint32_t neighbor_comm = node_data[i].community;

        if (neighbor_comm == curr_comm || neighbor_comm == best_comm) {
            continue;
        }
        // if (neighbor_comm == curr_comm) {
        //     continue;
        // }

        // aggregate count of nodes in new community (excluding current node)
        int agg_count_new = comm_data[neighbor_comm].agg_count;

        // total edge weight of incoming edges from new community
        float k_vc_new = node_to_comm_weights_final[i];
        // TODO: need to try moving this elsewhere
        // for (uint32_t j = offset; j < offset_next; j++) {
        //     uint32_t neigh = indices[j];
        //     // must include the self-edge here
        //     if (node_data[neigh].community == neighbor_comm || neigh == node) {
        //         k_vc_new += weights[j];
        //     }
        // }

        float delta = (k_vc_new - gamma * (float)(node_agg_count * agg_count_new)) - (k_vc_old - gamma * (float)(node_agg_count * (agg_count_old - node_agg_count)));

        if (delta > best_delta) {
            best_delta = delta;
            best_comm = neighbor_comm;
            // printf("Node: %d, Delta: %f, best_comm: %d\n", node, delta, best_comm);
        }
    }

    uint32_t comm_lo = curr_comm;
    uint32_t comm_hi = best_comm;
    if (best_comm < curr_comm) {
        comm_lo = best_comm;
        comm_hi = curr_comm;
    }

    if (best_comm != curr_comm) {
        if (atomicCAS(&(comm_locks[comm_lo]), 0, 1) == 0 && atomicCAS(&(comm_locks[comm_hi]), 0, 1) == 0) {
            // node_data[node].community = best_comm;
            // comm_data[best_comm].agg_count += node_agg_count;
            // comm_data[curr_comm].agg_count -= node_agg_count;
            node_moves[node] = best_comm;
            node_visited[node] = true;
            // *changed = true;
        }

        // if (best_delta > 1.0) {
        //     printf("Moving node: %d, to comm: %d, delta: %f\n", node, best_comm, best_delta);
        // }
        // node_moves[node] = best_comm;
        // *changed = true;
    } else {
        node_visited[node] = true;
    }
}

__global__ void has_zero_kernel(
    bool *node_visited,
    int vertex_count,
    bool *has_zero
) {
    unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= vertex_count) {
        return;
    }

    if (!node_visited[node]) {
        *has_zero = true;
    }
}

__global__ void edge_gather_new_neighbor_comm_weights_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    float *node_to_comm_weights,
    node_data_t *node_data,
    comm_data_t *comm_data,
    int *comm_locks,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma,
    uint32_t *node_moves,
    uint32_t *full_edge_list_u,
    uint32_t *full_edge_list_v
) {
    unsigned int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_idx >= edge_count) {
        return;
    }

    uint32_t node = full_edge_list_u[edge_idx];
    uint32_t neigh = full_edge_list_v[edge_idx];
    uint32_t weight = weights[edge_idx];

    uint32_t curr_comm = node_data[node].community;

    uint32_t n_offset = offsets[neigh];
    uint32_t n_offset_next = offsets[neigh + 1];

    for (int j = n_offset; j < n_offset_next; j++) {
        uint32_t n_neigh = indices[j];

        uint32_t n_comm = node_data[n_neigh].community;

        if (n_comm == curr_comm) {
            atomicAdd(&(node_to_comm_weights[j]), weight);
        }
    }
}

__global__ void gather_new_neighbor_comm_weights_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    float *node_to_comm_weights,
    node_data_t *node_data,
    comm_data_t *comm_data,
    int *comm_locks,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma,
    uint32_t *node_moves
) {
    unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= vertex_count) {
        return;
    }

    uint32_t curr_comm = node_data[node].community;
    // uint32_t best_comm = node_moves[node];

    // if (curr_comm == best_comm) {
    //     return;
    // }

    uint32_t offset = offsets[node];
    uint32_t offset_next = offsets[node + 1];

    // uint32_t node_edge_count = offset_next - offset;

    for (int i = offset; i < offset_next; i++) {
        uint32_t neigh = indices[i];
        float weight = weights[i];

        uint32_t n_offset = offsets[neigh];
        uint32_t n_offset_next = offsets[neigh + 1];

        // uint32_t prev_comm = orig_node_comms[neighbor];
        // uint32_t curr_comm = node_data[neighbor].community;

        // node_to_comm_comms[i] = curr_comm;
        // node_to_comm_weights[i] = weight;

        // trying to iterate the communities that neighbor is a neighbor of
        // bool found_old = false;
        // bool found_new = false;

        for (int j = n_offset; j < n_offset_next; j++) {
            uint32_t n_neigh = indices[j];

            uint32_t n_comm = node_data[n_neigh].community;

            if (n_comm == curr_comm) {
                atomicAdd(&(node_to_comm_weights[j]), weight);
            }

            // if (n_comm == curr_comm) {
            //     atomicAdd(&(node_to_comm_weights[j]), -weight);
            //     // found_old = true;
            // } else if (n_comm == best_comm) {
            //     atomicAdd(&(node_to_comm_weights[j]), weight);
            //     // found_new = true;
            // }

            // if (found_old && found_new) {
            //     break;
            // }
        }
    }
}

__global__ void apply_node_moves_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    uint32_t *node_agg_counts,
    comm_data_t *comm_data,
    int *comm_locks,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma,
    bool *changed,
    uint32_t *partition,
    uint32_t *node_moves,
    bool *node_visited
) {
    unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= vertex_count) {
        return;
    }
    uint32_t curr_comm = node_data[node].community;
    int node_agg_count = node_agg_counts[node];

    uint32_t best_comm = node_moves[node];

    if (best_comm != curr_comm) {
        node_data[node].community = best_comm;
        // comm_data[best_comm].agg_count += node_agg_count;
        // comm_data[curr_comm].agg_count -= node_agg_count;
        atomicAdd(&(comm_data[best_comm].agg_count), node_agg_count);
        atomicAdd(&(comm_data[curr_comm].agg_count), -node_agg_count);

        uint32_t offset = offsets[node];
        uint32_t offset_next = offsets[node + 1];

        // If you want to re-queue neighbors of a freshly moved node
        for (int i = offset; i < offset_next; i++) {
            uint32_t neighbor = indices[i];
            if (neighbor == node) {
                continue;   
            }

            node_visited[neighbor] = false;
        }
    }
}

// here we iterate over nodes and gather the comm-weight pairs for all the communities that a node has edges to
__global__ void gather_node_to_comm_comms_weights_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    uint32_t *node_to_comm_comms,
    float *node_to_comm_weights,
    node_data_t *node_data,
    comm_data_t *comm_data,
    int *comm_locks,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma,
    bool *changed,
    uint32_t *partition,
    uint32_t *orig_node_comms
) {
    unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= vertex_count) {
        return;
    }

    uint32_t offset = offsets[node];
    uint32_t offset_next = offsets[node + 1];

    // uint32_t node_edge_count = offset_next - offset;

    for (uint32_t i = offset; i < offset_next; i++) {
        uint32_t neighbor = indices[i];
        float weight = weights[i];

        // uint32_t prev_comm = orig_node_comms[neighbor];
        uint32_t curr_comm = node_data[neighbor].community;

        node_to_comm_comms[i] = curr_comm;
        node_to_comm_weights[i] = weight;

        // node_to_comm_comms[offset + (i + 1) % node_edge_count] = curr_comm;
        // node_to_comm_weights[offset + (i + 1) % node_edge_count] = weight;
    }
}

__global__ void scan_node_to_comm_comms_weights_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    uint32_t *node_to_comm_comms_sorted,
    float *node_to_comm_weights_sorted,
    node_data_t *node_data,
    comm_data_t *comm_data,
    int *comm_locks,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma,
    bool *changed,
    uint32_t *partition,
    uint32_t *orig_node_comms,
    uint32_t *node_to_comm_counts_final,
    uint32_t *node_to_comm_comms_final,
    float *node_to_comm_weights_final
) {
    unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= vertex_count) {
        return;
    }

    uint32_t offset = offsets[node];
    uint32_t offset_next = offsets[node + 1];

    uint32_t node_edge_count = offset_next - offset;

    if (node_edge_count == 0) {
        return;
    }

    uint32_t node_comm_count = 1;
    uint32_t curr_comm = node_to_comm_comms_sorted[offset];
    float weight_tot = 0.0f;

    for (uint32_t i = offset; i < offset_next; i++) {
        uint32_t next_comm = node_to_comm_comms_sorted[i];
        float weight = node_to_comm_weights_sorted[i];

        if (next_comm != curr_comm) {
            // save curr_comm and weight_tot here
            node_to_comm_comms_final[offset + node_comm_count - 1] = curr_comm;
            node_to_comm_weights_final[offset + node_comm_count - 1] = weight_tot;

            node_comm_count++;
            weight_tot = 0.0f;
        }

        weight_tot += weight;
        curr_comm = next_comm;
    }

    // save final curr_comm and weight_tot here
    node_to_comm_comms_final[offset + node_comm_count - 1] = curr_comm;
    node_to_comm_weights_final[offset + node_comm_count - 1] = weight_tot;

    // printf("Node %d comm count: %d\n", node, node_comm_count);

    node_to_comm_counts_final[node] = node_comm_count;
}

__global__ void create_partition_kernel(
    node_data_t *node_data,
    comm_data_t *comm_data,
    part_scan_data_t *part_scan_data,
    uint32_t *partition,
    int vertex_count,
    int comm_count
) {
    unsigned int node = threadIdx.x;

    uint32_t comm = node_data[node].community;
    uint32_t overall_offset = part_scan_data[comm].scanned_agg_count;
    uint32_t comm_offset = atomicAdd(&(part_scan_data[comm].curr_node_idx), 1);

    partition[overall_offset + comm_offset] = node;
}

// this gets the "well-connected" nodes within the partition we are refining
__global__ void refine_get_r_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    uint32_t *node_agg_counts,
    int vertex_count,
    int edge_count,
    float gamma,
    uint32_t *partition,
    uint32_t *partition_offsets,
    int partition_count,
    uint32_t *node_part,
    uint32_t *r_len,
    uint32_t *r,
    uint32_t *s_tots
) {
    unsigned int part_idx = threadIdx.x;

    uint32_t part_offset = partition_offsets[part_idx];
    uint32_t part_offset_next = partition_offsets[part_idx + 1];

    // printf("Partition %d: %d - %d\n", part_idx, part_offset, part_offset_next);

    uint32_t s_tot = 0;

    for (int i = part_offset; i < part_offset_next; i++) {
        uint32_t node = partition[i];
        // TODO: may need to mark that this node is a member of this partition, so that we can
        // tell when we iterate the nodes

        s_tot += node_agg_counts[node];

        node_part[node] = part_idx;
    }

    // need to sync threads since one partition will see node_part data of other partitions
    __syncthreads();

    s_tots[part_idx] = s_tot;

    int local_r_len = 0;

    for (int i = part_offset; i < part_offset_next; i++) {
        uint32_t node = partition[i];
        
        uint32_t v_tot = node_agg_counts[node];

        uint32_t v_in = 0;

        uint32_t v_offset = offsets[node];
        uint32_t v_offset_next = offsets[node + 1];

        for (int j = v_offset; j < v_offset_next; j++) {
            uint32_t neighbor = indices[j];

            // no self-edges allowed here
            if (neighbor == node) {
                continue;
            }

            float weight = weights[j];

            uint32_t neighbor_part_idx = node_part[neighbor];

            if (neighbor_part_idx == part_idx) {
                v_in += weight;
            }
        }

        if (v_in >= gamma * (float)(v_tot * (s_tot - v_tot))) {
            r[part_offset + local_r_len] = node;
            local_r_len++;
        }
    }

    r_len[part_idx] = local_r_len;
}

// this parallelizes at the node level
__global__ void set_initial_refined_comm_in_edge_weights_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    int vertex_count,
    int edge_count,
    float gamma,
    uint32_t *partition,
    uint32_t *partition_offsets,
    int partition_count,
    uint32_t *node_part,
    uint32_t *r_len,
    uint32_t *r,
    uint32_t *s_tots,
    uint32_t *node_refined_comms,
    float *refined_comm_in_edge_weights,
    uint32_t *refined_comm_agg_counts
) {
    unsigned int node = threadIdx.x;
    uint32_t comm = node_refined_comms[node];
    uint32_t part = node_part[node];

    uint32_t v_offset = offsets[node];
    uint32_t v_offset_next = offsets[node + 1];

    float tot_weight = 0.0f;

    for (int j = v_offset; j < v_offset_next; j++) {
        uint32_t neighbor = indices[j];
        float weight = weights[j];
        uint32_t neighbor_part = node_part[neighbor];

        if (neighbor_part == part) {
            tot_weight += weight;
        }
    }

    refined_comm_in_edge_weights[comm] = tot_weight;
}

__global__ void refine_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    uint32_t *node_agg_counts,
    int vertex_count,
    int edge_count,
    float gamma,
    uint32_t *partition,
    uint32_t *partition_offsets,
    int partition_count,
    uint32_t *node_part,
    uint32_t *r_len,
    uint32_t *r,
    uint32_t *s_tots,
    uint32_t *node_refined_comms,
    float *refined_comm_in_edge_weights,
    uint32_t *refined_comm_agg_counts
) {
    unsigned int part_idx = threadIdx.x;
    uint32_t part_offset = partition_offsets[part_idx];
    // uint32_t part_offset_next = partition_offsets[part_idx + 1];
    uint32_t local_r_len = r_len[part_idx];
    uint32_t s_tot = s_tots[part_idx];

    // this iterates over R (subset of nodes in the partition being refined that are "well-connected")
    for (int i = part_offset; i < part_offset + local_r_len; i++) {
        uint32_t node = r[i];
        uint32_t curr_comm = node_refined_comms[node];
        uint32_t node_agg_count = node_agg_counts[i];

        uint32_t agg_count_old = refined_comm_agg_counts[curr_comm];

        bool is_in_singleton_community = true;
        uint32_t v_offset = offsets[node];
        uint32_t v_offset_next = offsets[node + 1];

        for (int j = v_offset; j < v_offset_next; j++) {
            uint32_t neighbor = indices[j];

            // skip self-edges
            if (neighbor == node) {
                continue;
            }

            uint32_t neighbor_comm = node_refined_comms[neighbor];
            // TODO: could be smarter about checking if this node is in a singleton partition
            if (curr_comm == neighbor_comm) {
                is_in_singleton_community = false;
                break;
            }
        }

        if (!is_in_singleton_community) {
            continue;
        }

        float k_vc_old = 0.0;
        for (uint32_t j = v_offset; j < v_offset_next; j++) {
            uint32_t neigh = indices[j];
            if (node_refined_comms[neigh] == curr_comm) {
                k_vc_old += weights[j];
            }
        }

        uint32_t best_comm = curr_comm;
        float best_delta = 0;

        // this iterates over the neighbors of the node we are currently looking at
        for (int j = v_offset; j < v_offset_next; j++) {
            uint32_t neighbor = indices[j];

            // skip self-edges
            if (neighbor == node) {
                continue;
            }

            uint32_t neighbor_part_idx = node_part[neighbor];
            // only consider neighbors within this partition
            if (neighbor_part_idx != part_idx) {
                continue;
            }

            uint32_t neighbor_comm = node_refined_comms[neighbor];

            uint32_t c_tot = refined_comm_agg_counts[neighbor_comm];
            uint32_t c_in = refined_comm_in_edge_weights[neighbor_comm];

            if (c_in >= gamma * (float)(c_tot * (s_tot - c_tot))) {
                float k_vc_new = 0.0;
                for (uint32_t k = v_offset; k < v_offset_next; k++) {
                    uint32_t neigh = indices[k];
                    if (node_refined_comms[neigh] == neighbor_comm) {
                        k_vc_new += weights[k];
                    }
                }

                uint32_t agg_count_new = refined_comm_agg_counts[neighbor_comm];

                float delta = (k_vc_new - gamma * (float)(node_agg_count * agg_count_new)) - (k_vc_old - gamma * (float)(node_agg_count * (agg_count_old - node_agg_count)));

                if (delta > best_delta) {
                    best_delta = delta;
                    best_comm = neighbor_comm;
                    // printf("Node: %d, Delta: %f, best_comm: %d\n", node, delta, best_comm);
                }
            }
        }

        // TODO: move node to best community
        // if we actually perform a move, need to upate all of the following:
        // uint32_t *node_refined_comms,
        // float *refined_comm_in_edge_weights,
        // uint32_t *refined_comm_agg_counts

        if (best_comm != curr_comm) {
            // printf("Found move for node: %d\n", node);

            refined_comm_agg_counts[best_comm] += node_agg_count;
            refined_comm_agg_counts[curr_comm] -= node_agg_count;

            // TODO: refined_comm_in_edge_weights
            // need to reduce the refined edge weight of best_comm by the amount of edge weight from nodes in best_comm
            // to the new node, given that the new node will now be part of best_comm
            // - we can simultaneously add the edge weight to nodes outside of best_comm (making sure to avoid including self-edge weight)
            // and ensuring that the edge weight we are adding is to nodes within the current partition (skip if not)

            float refined_comm_in_edge_weight_change = 0.0f;
            for (int j = v_offset; j < v_offset_next; j++) {
                uint32_t neighbor = indices[j];
                if (neighbor == node) {
                    continue;
                }

                float weight = weights[j];

                uint32_t neighbor_part_idx = node_part[neighbor];
                uint32_t neighbor_comm = node_refined_comms[neighbor];

                if (neighbor_part_idx == part_idx) {
                    if (neighbor_comm == best_comm) {
                        refined_comm_in_edge_weight_change -= weight;
                    } else {
                        refined_comm_in_edge_weight_change += weight;
                    }
                }
            }

            // since this is a singleton community we are moving out of (not actually needed)
            refined_comm_in_edge_weights[curr_comm] = 0.0f;

            refined_comm_in_edge_weights[best_comm] += refined_comm_in_edge_weight_change;
            
            // IMPORTANT: this must happen after updating refined_comm_in_edge_weights
            // because we do not want to involve self-edges in the above computation
            node_refined_comms[node] = best_comm;

            // atomicAdd(&(comm_data[best_comm].agg_count), 1);
            // atomicAdd(&(comm_data[curr_comm].agg_count), -1);
        }
    }
}

__global__ void count_partitions_kernel(
    comm_data_t *comm_data,
    int comm_count,
    uint32_t *partition_count
) {
    unsigned int comm = blockIdx.x * blockDim.x + threadIdx.x;

    if (comm >= comm_count) {
        return;
    }

    uint32_t agg_count = comm_data[comm].agg_count;

    if (agg_count > 0) {
        atomicAdd(partition_count, 1);
    }
}

__global__ void cpm_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma,
    float *cpm_comm_internal_sums
) {
    unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= vertex_count) {
        return;
    }

    uint32_t curr_comm = node_data[node].community;

    uint32_t offset = offsets[node];
    uint32_t offset_next = offsets[node + 1];

    float tot = 0.0f;

    for (uint32_t i = offset; i < offset_next; i++) {
        uint32_t neighbor = indices[i];
        float weight = weights[i];

        uint32_t neighbor_comm = node_data[neighbor].community;

        if (curr_comm != neighbor_comm) {
            continue;
        }

        if (neighbor == node) {
            // everything else is double counted, so we need to double if it is a self-edge
            tot += 2.0 * weight;
        } else {
            tot += weight;
        }
    }

    atomicAdd(&(cpm_comm_internal_sums[curr_comm]), tot);
}

template <typename T>
T* allocate_and_copy_to_device(T* data_host, int len) {
    T* data_device;

    int size = len * sizeof(T);
    cudaMalloc((void**)&data_device, size);
    cudaMemcpy(data_device, data_host, size, cudaMemcpyHostToDevice);

    return data_device;
}

template <typename T>
void copy_to_device(T* data_host, T* data_device, int len) {
    int size = len * sizeof(T);
    cudaMemcpy(data_device, data_host, size, cudaMemcpyHostToDevice);
}

template <typename T>
void copy_from_device(T* data_host, T* data_device, int len) {
    int size = len * sizeof(T);
    cudaMemcpy(data_host, data_device, size, cudaMemcpyDeviceToHost);
}

void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

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
) {
    // each thread of the cuda kernel considers one node and attempts to greedily increase the CPM
    // by moving it to the best neighboring community

    // the threads should be considering nodes in a semi-random order though.
    // will it be better to give them an array of random indices to access,
    // - or should we reorder the data structure to ensure a warp coalesces global memory accesses?

    bool *changed = (bool *)malloc(sizeof(bool));
    bool *has_zero = (bool *)malloc(sizeof(bool));
    // cudaMemset((void *)binsDevice, 0, binsSize);

    int *comm_locks = (int *)malloc(comm_count * sizeof(int));
    memset(comm_locks, 0, comm_count * sizeof(float));

    // TODO: improve this and make sure it is initialized correctly on subsequent iterations
    uint32_t *orig_node_comms = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));
    for (int i = 0; i < vertex_count; i++) {
        orig_node_comms[i] = node_data[i].community;
    }

    uint32_t *node_moves = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));

    for (int i = 0; i < vertex_count; i++) {
        node_moves[i] = node_data[i].community;
    }

    uint32_t *node_to_comm_comms = (uint32_t *)malloc(edge_count * sizeof(uint32_t));
    float *node_to_comm_weights = (float *)malloc(edge_count * sizeof(float));

    uint32_t *node_moves_device = allocate_and_copy_to_device(node_moves, vertex_count);

    uint32_t *offsets_device = allocate_and_copy_to_device(offsets, vertex_count + 1);
    uint32_t *indices_device = allocate_and_copy_to_device(indices, edge_count);
    float *weights_device = allocate_and_copy_to_device(weights, edge_count);
    uint32_t *full_edge_list_u_device = allocate_and_copy_to_device(full_edge_list_u, edge_count);
    uint32_t *full_edge_list_v_device = allocate_and_copy_to_device(full_edge_list_v, edge_count);

    // these can technically be uninitialized since we initialize them in a kernel
    uint32_t *node_to_comm_comms_device = allocate_and_copy_to_device(node_to_comm_comms, edge_count);
    float *node_to_comm_weights_device = allocate_and_copy_to_device(node_to_comm_weights, edge_count);

    int *comm_locks_device = allocate_and_copy_to_device(comm_locks, comm_count);
    node_data_t *node_data_device = allocate_and_copy_to_device(node_data, vertex_count);
    uint32_t *node_agg_counts_device = allocate_and_copy_to_device(node_agg_counts, vertex_count);
    comm_data_t *comm_data_device = allocate_and_copy_to_device(comm_data, comm_count);
    bool *changed_device = allocate_and_copy_to_device(changed, 1);
    bool *has_zero_device = allocate_and_copy_to_device(has_zero, 1);
    uint32_t *orig_node_comms_device = allocate_and_copy_to_device(orig_node_comms, vertex_count);

    uint32_t *partition = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));
    uint32_t *partition_device = allocate_and_copy_to_device(partition, vertex_count);

    uint32_t *node_part = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));
    uint32_t *node_part_device = allocate_and_copy_to_device(node_part, vertex_count);

    uint32_t *r = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));
    uint32_t *r_device = allocate_and_copy_to_device(r, vertex_count);


    uint32_t *node_refined_comms = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));
    float *refined_comm_in_edge_weights = (float *)malloc(vertex_count * sizeof(float));
    uint32_t *refined_comm_agg_counts = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));


    uint32_t *node_to_comm_counts_final = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));

    for (int i = 0; i < vertex_count; i++) {
        node_to_comm_counts_final[i] = offsets[i + 1] - offsets[i];
    }

    uint32_t *node_to_comm_comms_final = (uint32_t *)malloc(edge_count * sizeof(uint32_t));
    float *node_to_comm_weights_final = (float *)malloc(edge_count * sizeof(float));

    uint32_t *node_to_comm_counts_final_device = allocate_and_copy_to_device(node_to_comm_counts_final, vertex_count);
    uint32_t *node_to_comm_comms_final_device = allocate_and_copy_to_device(indices, edge_count);
    float *node_to_comm_weights_final_device = allocate_and_copy_to_device(weights, edge_count);

    // this is for computing CPM on the GPU at the end
    float *cpm_comm_internal_sums = (float *)malloc(comm_count * sizeof(float));
    memset(cpm_comm_internal_sums, 0, comm_count * sizeof(float));
    float *cpm_comm_internal_sums_device = allocate_and_copy_to_device(cpm_comm_internal_sums, comm_count);


    uint32_t *partition_count_host = (uint32_t *)malloc(sizeof(uint32_t));
    *partition_count_host = 0;
    uint32_t *partition_count_device = allocate_and_copy_to_device(partition_count_host, 1);

    bool *node_visited = (bool *)malloc(vertex_count * sizeof(bool));
    memset(node_visited, 0, vertex_count * sizeof(bool));
    bool *node_visited_device = allocate_and_copy_to_device(node_visited, vertex_count);

    for (int i = 0; i < vertex_count; i++) {
        node_refined_comms[i] = i;
        // TODO: refined_comm_in_edge_weights -- need to initialize this correctly
        refined_comm_in_edge_weights[i] = 0;
        // IMPORTANT: must make sure we get the agg count here from the node
        refined_comm_agg_counts[i] = node_agg_counts[i];
    }

    uint32_t *node_refined_comms_device = allocate_and_copy_to_device(node_refined_comms, vertex_count);
    float *refined_comm_in_edge_weights_device = allocate_and_copy_to_device(refined_comm_in_edge_weights, vertex_count);
    uint32_t *refined_comm_agg_counts_device = allocate_and_copy_to_device(refined_comm_agg_counts, vertex_count);

    printf("move_nodes_fast starting, checking for CUDA error...\n");
    checkCudaError();

    int block_size = 1024;

    int grid_size = vertex_count / block_size;
    if (vertex_count % block_size != 0) {
        grid_size += 1;
    }

    int edge_grid_size = edge_count / block_size;
    if (edge_count % block_size != 0) {
        edge_grid_size += 1;
    }

    printf("Vertex count: %d, Block size: %d, grid size: %d, edge grid size: %d\n", vertex_count, block_size, grid_size, edge_grid_size);

    dim3 edge_dim_grid(edge_grid_size);
    dim3 dim_grid(grid_size);
 	dim3 dim_block(block_size);

    int move_nodes_fast_iter = 0;

    float *d_random;
    curandState *d_state;

    cudaMalloc(&d_random, vertex_count * sizeof(float));
    cudaMalloc(&d_state, vertex_count * sizeof(curandState));

    init_rng<<<dim_grid, dim_block>>>(d_state, 1234);

    uint32_t* node_to_comm_comms_sorted_device;
    int node_to_comm_comms_size = edge_count * sizeof(uint32_t);
    cudaMalloc((void**)&node_to_comm_comms_sorted_device, node_to_comm_comms_size);

    float* node_to_comm_weights_sorted_device;
    int node_to_comm_weights_size = edge_count * sizeof(float);
    cudaMalloc((void**)&node_to_comm_weights_sorted_device, node_to_comm_weights_size);

    // void *d_temp_storage = nullptr;
    // size_t temp_storage_bytes = 0;
    // cub::DeviceSegmentedRadixSort::SortPairs(
    //     d_temp_storage, temp_storage_bytes,
    //     node_to_comm_comms_device,
    //     node_to_comm_comms_sorted_device,
    //     node_to_comm_weights_device,
    //     node_to_comm_weights_sorted_device,
    //     edge_count, vertex_count, offsets_device, offsets_device + 1);

    // // Allocate temporary storage
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);

    uint32_t prev_partition_count = comm_count;

    while (true) {
        generate_random<<<dim_grid, dim_block>>>(d_random, d_state, vertex_count);

        move_nodes_fast_kernel <<<dim_grid, dim_block>>> (
            offsets_device,
            indices_device,
            weights_device,
            node_data_device,
            node_agg_counts_device,
            comm_data_device,
            comm_locks_device,
            vertex_count,
            edge_count,
            comm_count,
            gamma,
            changed_device,
            partition_device,
            node_moves_device,
            node_to_comm_counts_final_device,
            node_to_comm_comms_final_device,
            node_to_comm_weights_final_device,
            d_random,
            node_visited_device
        );
        cudaDeviceSynchronize();

        has_zero_kernel <<<dim_grid, dim_block>>> (
            node_visited_device,
            vertex_count,
            has_zero_device
        );
        cudaDeviceSynchronize();

        copy_from_device(has_zero, has_zero_device, 1);

        if (!*has_zero) {
            printf("No more unvisited nodes!\n");
            break;
        }

        cudaMemset((void *)has_zero_device, 0, sizeof(bool));

        // copy_from_device(changed, changed_device, 1);

        // if (!*changed) {
        //     break;
        // }

        apply_node_moves_kernel <<<dim_grid, dim_block>>> (
            offsets_device,
            indices_device,
            weights_device,
            node_data_device,
            node_agg_counts_device,
            comm_data_device,
            comm_locks_device,
            vertex_count,
            edge_count,
            comm_count,
            gamma,
            changed_device,
            partition_device,
            node_moves_device,
            node_visited_device
        );
        cudaDeviceSynchronize();

        // HERE WE COULD PARALLELIZE BY EDGES, updating node_to_comm_weights_final_device
        // FOR THE DST NODE, modifying atomically both the old comm and the new comm weight that the 

        cudaMemset((void *)node_to_comm_weights_final_device, 0, edge_count * sizeof(float));

        edge_gather_new_neighbor_comm_weights_kernel <<<edge_dim_grid, dim_block>>> (
            offsets_device,
            indices_device,
            weights_device,
            node_to_comm_weights_final_device,
            node_data_device,
            comm_data_device,
            comm_locks_device,
            vertex_count,
            edge_count,
            comm_count,
            gamma,
            node_moves_device,
            full_edge_list_u_device,
            full_edge_list_v_device
        );
        cudaDeviceSynchronize();
        checkCudaError();

        // gather_new_neighbor_comm_weights_kernel <<<dim_grid, dim_block>>> (
        //     offsets_device,
        //     indices_device,
        //     weights_device,
        //     node_to_comm_weights_final_device,
        //     node_data_device,
        //     comm_data_device,
        //     comm_locks_device,
        //     vertex_count,
        //     edge_count,
        //     comm_count,
        //     gamma,
        //     node_moves_device
        // );
        // cudaDeviceSynchronize();
        // checkCudaError();

        // gather_node_to_comm_comms_weights_kernel <<<dim_grid, dim_block>>> (
        //     offsets_device,
        //     indices_device,
        //     weights_device,
        //     node_to_comm_comms_device,
        //     node_to_comm_weights_device,
        //     node_data_device,
        //     comm_data_device,
        //     comm_locks_device,
        //     vertex_count,
        //     edge_count,
        //     comm_count,
        //     gamma,
        //     changed_device,
        //     partition_device,
        //     node_moves_device
        // );
        // cudaDeviceSynchronize();
        // checkCudaError();

        // Run sorting operation
        // cub::DeviceSegmentedRadixSort::SortPairs(
        //     d_temp_storage, temp_storage_bytes,
        //     node_to_comm_comms_device,
        //     node_to_comm_comms_sorted_device,
        //     node_to_comm_weights_device,
        //     node_to_comm_weights_sorted_device,
        //     edge_count, vertex_count, offsets_device, offsets_device + 1);

        // scan_node_to_comm_comms_weights_kernel <<<dim_grid, dim_block>>> (
        //     offsets_device,
        //     indices_device,
        //     weights_device,
        //     node_to_comm_comms_sorted_device,
        //     node_to_comm_weights_sorted_device,
        //     node_data_device,
        //     comm_data_device,
        //     comm_locks_device,
        //     vertex_count,
        //     edge_count,
        //     comm_count,
        //     gamma,
        //     changed_device,
        //     partition_device,
        //     node_moves_device,
        //     node_to_comm_counts_final_device,
        //     node_to_comm_comms_final_device,
        //     node_to_comm_weights_final_device
        // );
        // cudaDeviceSynchronize();
        // checkCudaError();

        // reset changed
        cudaMemset((void *)changed_device, 0, sizeof(bool));

        // reset comm locks before next iteration
        int comm_locks_size = comm_count * sizeof(int);
        cudaMemset((void *)comm_locks_device, 0, comm_locks_size);

        move_nodes_fast_iter++;

        count_partitions_kernel <<<dim_grid, dim_block>>> (
            comm_data_device,
            comm_count,
            partition_count_device
        );
        cudaDeviceSynchronize();
        checkCudaError();

        copy_from_device(partition_count_host, partition_count_device, 1);
        cudaMemset((void *)partition_count_device, 0, sizeof(uint32_t));

        prev_partition_count = *partition_count_host;

        printf("Move nodes fast iter: %d, partition count: %d\n", move_nodes_fast_iter, prev_partition_count);
        if (move_nodes_fast_iter == 10) {
            break;
        }
    }

    // copy_from_device(node_data, node_data_device, vertex_count);
    // copy_from_device(node_to_comm_counts_final, node_to_comm_counts_final_device, vertex_count);
    // copy_from_device(node_to_comm_comms_final, node_to_comm_comms_final_device, edge_count);
    // copy_from_device(node_to_comm_weights_final, node_to_comm_weights_final_device, edge_count);

    // for (int node = 0; node < vertex_count; node++) {
    //     uint32_t offset = offsets[node];
    //     uint32_t comm_count = node_to_comm_counts_final[node];
    //     uint32_t offset_next = offset + comm_count;
    //     float weight_tot = 0.0f;

    //     for (int i = offset; i < offset_next; i++) {
    //         uint32_t neigh = indices[i];
    //         uint32_t comm = node_data[neigh].community;
    //         float weight = node_to_comm_weights_final[i];

    //         weight_tot += weight;
    //         printf("Node %d, comm: %d, weight: %f\n", node, comm, weight);
    //     }

    //     printf("Node %d, comm_count: %d, weight_tot: %f\n", node, comm_count, weight_tot);
    // }

    printf("move_nodes_fast complete, checking for CUDA error...\n");
    checkCudaError();

    copy_from_device(comm_data, comm_data_device, comm_count);

    int partition_count = prev_partition_count;
    // int partition_count = 0;
    // for (int i = 0; i < comm_count; i++) {
    //     uint32_t agg_count = comm_data[i].agg_count;
    //     if (agg_count > 0) {
    //         // printf("Partition %u count: %u\n", i, agg_count);
    //         partition_count++;
    //     }
    // }

    printf("\nPartition count after move_nodes_fast: %d\n", partition_count);

    // copy_from_device(node_data, node_data_device, vertex_count);
    // copy_from_device(comm_data, comm_data_device, comm_count);

    cpm_kernel <<<dim_grid, dim_block>>> (
        offsets_device,
        indices_device,
        weights_device,
        node_data_device,
        vertex_count,
        edge_count,
        comm_count,
        gamma,
        cpm_comm_internal_sums_device
    );

    copy_from_device(cpm_comm_internal_sums, cpm_comm_internal_sums_device, vertex_count);

    float cpm_tot = 0.0f;

    for (int i = 0; i < comm_count; i++) {
        uint32_t agg_count = comm_data[i].agg_count;
        if (agg_count > 0) {
            float internal_weight_sum = cpm_comm_internal_sums[i];
            float contribution = internal_weight_sum - gamma * ((float)(agg_count * (agg_count - 1)));
            // if (contribution < 0) {
            //     printf("Partition %u count: %u, %f\n", i, agg_count, internal_weight_sum);
            // }
            cpm_tot += contribution;
            // should there be / 2.0 here, as in agg_count * (agg_count - 1) / 2.0?
            // I believe not, since everything in internal_weight_sum is double counted
            // so we should also double count the max possible edges
        }
    }

    float cpm = cpm_tot;

    printf("CPM: %f\n", cpm);

    return;

    // TODO: we can get away with making this smaller, but it changes the indexing approach in the next kernel
    part_scan_data_t *part_scan_data = (part_scan_data_t *)malloc(comm_count * sizeof(part_scan_data_t));
    uint32_t *partition_offsets = (uint32_t *)malloc((partition_count + 1) * sizeof(uint32_t));

    uint32_t scan_idx = 0;
    partition_count = 0;
    for (int i = 0; i < comm_count; i++) {
        part_scan_data_t p = { .scanned_agg_count = scan_idx, .curr_node_idx = 0 };
        part_scan_data[i] = p;

        uint32_t agg_count = comm_data[i].agg_count;
        if (agg_count > 0) {
            partition_offsets[partition_count] = scan_idx;
            partition_count++;
            scan_idx += agg_count;
        }
    }
    partition_offsets[partition_count] = scan_idx;

    uint32_t *r_len = (uint32_t *)malloc(partition_count * sizeof(uint32_t));
    uint32_t *r_len_device = allocate_and_copy_to_device(r_len, partition_count);

    uint32_t *s_tots = (uint32_t *)malloc(partition_count * sizeof(uint32_t));
    uint32_t *s_tots_device = allocate_and_copy_to_device(s_tots, partition_count);

    part_scan_data_t *part_scan_data_device = allocate_and_copy_to_device(part_scan_data, comm_count);

    create_partition_kernel <<<dim_grid, dim_block>>> (
        node_data_device,
        comm_data_device,
        part_scan_data_device,
        partition_device,
        vertex_count,
        comm_count
    );
    cudaDeviceSynchronize();

    dim3 refine_dim_block(partition_count);

    uint32_t *partition_offsets_device = allocate_and_copy_to_device(partition_offsets, partition_count + 1);

    refine_get_r_kernel <<<dim_grid, refine_dim_block>>> (
        offsets_device,
        indices_device,
        weights_device,
        node_data_device,
        node_agg_counts_device,
        vertex_count,
        edge_count,
        gamma,
        partition_device,
        partition_offsets_device, 
        partition_count,
        node_part_device,
        r_len_device,
        r_device,
        s_tots_device
    );
    cudaDeviceSynchronize();

    set_initial_refined_comm_in_edge_weights_kernel <<<dim_grid, dim_block>>> (
        offsets_device,
        indices_device,
        weights_device,
        node_data_device,
        vertex_count,
        edge_count,
        gamma,
        partition_device,
        partition_offsets_device, 
        partition_count,
        node_part_device,
        r_len_device,
        r_device,
        s_tots_device,
        node_refined_comms_device,
        refined_comm_in_edge_weights_device,
        refined_comm_agg_counts_device
    );
    cudaDeviceSynchronize();

    refine_kernel <<<dim_grid, refine_dim_block>>> (
        offsets_device,
        indices_device,
        weights_device,
        node_data_device,
        node_agg_counts_device,
        vertex_count,
        edge_count,
        gamma,
        partition_device,
        partition_offsets_device, 
        partition_count,
        node_part_device,
        r_len_device,
        r_device,
        s_tots_device,
        node_refined_comms_device,
        refined_comm_in_edge_weights_device,
        refined_comm_agg_counts_device
    );
    cudaDeviceSynchronize();
    checkCudaError();

    copy_from_device(node_refined_comms, node_refined_comms_device, vertex_count);
    copy_from_device(refined_comm_agg_counts, refined_comm_agg_counts_device, vertex_count);

    int refined_comm_count = 0;
    for (int i = 0; i < vertex_count; i++) {
        uint32_t agg_count = refined_comm_agg_counts[i];
        if (agg_count > 0) {
            refined_comm_count++;
        }
    }

    printf("\nRefined comm count: %d\n", refined_comm_count);

    // copy_from_device(refined_comm_in_edge_weights, refined_comm_in_edge_weights_device, vertex_count);

    // for (int i = 0; i < vertex_count; i++) {
    //     printf("Refined comm in edge weight %d: %f\n", i, refined_comm_in_edge_weights[i]);
    // }

    // copy_from_device(r_len, r_len_device, partition_count);

    // copy_from_device(partition, partition_device, vertex_count);

    // printf("\n---- Partition count %d -----\n\n", partition_count);

    // for (int i = 0; i < partition_count; i++) {
    //     printf("R len for partition %d: %d\n", i, r_len[i]);
    // }

    // printf("\n\n");

    // for (int i = 0; i < partition_count; i++) {
    //     uint32_t part_start = partition_offsets[i];
    //     uint32_t part_end = partition_offsets[i + 1];
    //     for (int j = part_start; j < part_end; j++) {
    //         printf("Partition %d: %d\n", i, partition[j]);
    //     }
    // }

    copy_from_device(offsets, offsets_device, vertex_count + 1);
    copy_from_device(indices, indices_device, edge_count);
    copy_from_device(weights, weights_device, edge_count);
    copy_from_device(node_data, node_data_device, vertex_count);
    copy_from_device(comm_data, comm_data_device, comm_count);

    cudaFree(offsets_device);
    cudaFree(indices_device);
    cudaFree(weights_device);
    cudaFree(node_data_device);
    cudaFree(comm_data_device);
}
