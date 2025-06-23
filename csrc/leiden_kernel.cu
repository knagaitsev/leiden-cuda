#include <cuda_runtime.h>
#include <iostream>

typedef struct node_data {
    uint32_t community;
    uint32_t move_candidate;
    uint32_t agg_count;
} node_data_t;

typedef struct comm_data {
    uint32_t agg_count;
} comm_data_t;

typedef struct part_scan_data {
    uint32_t scanned_agg_count;
    uint32_t curr_node_idx;
} part_scan_data_t;

__global__ void add_kernel(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

extern "C" void launch_add_kernel(float* a, float* b, float* c, int N) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// parallelized at node level
// - we should also try parallelizing at edge level
__global__ void gather_move_candidates_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    comm_data_t *comm_data,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma
) {
    unsigned int node = threadIdx.x;
    
    // communities[threadIdx.x] = 1;
    uint32_t offset = offsets[node];
    uint32_t offset_next = offsets[node + 1];

    uint32_t curr_comm = node_data[node].community;

    uint32_t best_comm = curr_comm;
    float best_delta = 0.0f;

    uint32_t node_edge_count = offset_next - offset;

    // aggregate count of nodes in old community (including current node)
    int agg_count_old = comm_data[curr_comm].agg_count;

    // aggregate count of current node
    int node_agg_count = node_data[node].agg_count;

    // total edge weight of incoming edges from old community
    float k_vc_old = 0.0;
    for (uint32_t i = offset; i < offset_next; i++) {
        uint32_t neigh = indices[i];
        if (node_data[neigh].community == curr_comm) {
            k_vc_old += weights[i];
        }
    }

    for (uint32_t i = offset; i < offset_next; i++) {
        uint32_t neighbor = indices[i];
        float weight = weights[i];

        uint32_t neighbor_comm = node_data[neighbor].community;

        if (neighbor_comm == curr_comm || neighbor_comm == best_comm) {
            continue;
        }

        // aggregate count of nodes in new community (excluding current node)
        int agg_count_new = comm_data[neighbor_comm].agg_count;

        // total edge weight of incoming edges from new community
        float k_vc_new = 0.0;
        // TODO: need to try moving this elsewhere
        for (uint32_t j = offset; j < offset_next; j++) {
            uint32_t neigh = indices[j];
            if (node_data[neigh].community == neighbor_comm) {
                k_vc_new += weights[j];
            }
        }

        float delta = (k_vc_new - gamma * (float)(node_agg_count * agg_count_new)) - (k_vc_old - gamma * (float)(node_agg_count * (agg_count_old - node_agg_count)));

        if (delta > best_delta) {
            // printf("Node: %d, Delta: %f, best_comm: %d\n", node, delta, best_comm);
            best_delta = delta;
            best_comm = neighbor_comm;
        }
    }

    if (best_comm != curr_comm) {
        node_data[node].move_candidate = best_comm;
        // if (atomicCAS(&comm_data[best_comm]))
    }
}

// two approaches to doing move_nodes_fast: parallelizing at node level is below
// - another option is parallelizing at edge level, letting each thread consider an edge
__global__ void move_nodes_fast_kernel(
    uint32_t *offsets,
    uint32_t *indices,
    float *weights,
    node_data_t *node_data,
    comm_data_t *comm_data,
    int vertex_count,
    int edge_count,
    int comm_count,
    float gamma,
    bool *changed,
    uint32_t *partition
) {
    unsigned int node = threadIdx.x;

    uint32_t offset = offsets[node];
    uint32_t offset_next = offsets[node + 1];

    while (true) {
        uint32_t curr_comm = node_data[node].community;

        uint32_t best_comm = curr_comm;
        float best_delta = 0.0f;

        uint32_t node_edge_count = offset_next - offset;

        // aggregate count of nodes in old community (including current node)
        int agg_count_old = comm_data[curr_comm].agg_count;

        // aggregate count of current node
        int node_agg_count = node_data[node].agg_count;

        // total edge weight of incoming edges from old community
        float k_vc_old = 0.0;
        for (uint32_t i = offset; i < offset_next; i++) {
            uint32_t neigh = indices[i];
            if (node_data[neigh].community == curr_comm) {
                k_vc_old += weights[i];
            }
        }

        for (uint32_t i = offset; i < offset_next; i++) {
            uint32_t neighbor = indices[i];
            float weight = weights[i];

            uint32_t neighbor_comm = node_data[neighbor].community;

            if (neighbor_comm == curr_comm || neighbor_comm == best_comm) {
                continue;
            }

            // aggregate count of nodes in new community (excluding current node)
            int agg_count_new = comm_data[neighbor_comm].agg_count;

            // total edge weight of incoming edges from new community
            float k_vc_new = 0.0;
            // TODO: need to try moving this elsewhere
            for (uint32_t j = offset; j < offset_next; j++) {
                uint32_t neigh = indices[j];
                if (node_data[neigh].community == neighbor_comm) {
                    k_vc_new += weights[j];
                }
            }

            float delta = (k_vc_new - gamma * (float)(node_agg_count * agg_count_new)) - (k_vc_old - gamma * (float)(node_agg_count * (agg_count_old - node_agg_count)));

            if (delta > best_delta) {
                // printf("Node: %d, Delta: %f, best_comm: %d\n", node, delta, best_comm);
                best_delta = delta;
                best_comm = neighbor_comm;
            }
        }

        if (best_comm != curr_comm) {
            node_data[node].community = best_comm;
            atomicAdd(&(comm_data[best_comm].agg_count), 1);
            atomicAdd(&(comm_data[curr_comm].agg_count), -1);

            *changed = true;
        }

        // IMPORTANT: this currently assumes all the threads are in one block
        __syncthreads();

        if (!*changed) {
            break;
        }

        __syncthreads();

        *changed = false;
    }
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
    int vertex_count,
    int edge_count,
    float gamma,
    uint32_t *partition,
    uint32_t *partition_offsets,
    int partition_count,
    uint32_t *node_part,
    uint32_t *r_len,
    uint32_t *r
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

        s_tot += node_data[node].agg_count;

        node_part[node] = part_idx;
    }

    // need to sync threads since one partition will see node_part data of other partitions
    __syncthreads();

    int local_r_len = 0;

    for (int i = part_offset; i < part_offset_next; i++) {
        uint32_t node = partition[i];
        
        uint32_t v_tot = node_data[node].agg_count;

        uint32_t v_in = 0;

        uint32_t v_offset = offsets[node];
        uint32_t v_offset_next = offsets[node + 1];

        for (int j = v_offset; j < v_offset_next; j++) {
            uint32_t neighbor = indices[j];
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
) {
    // each thread of the cuda kernel considers one node and attempts to greedily increase the CPM
    // by moving it to the best neighboring community

    // the threads should be considering nodes in a semi-random order though.
    // will it be better to give them an array of random indices to access,
    // - or should we reorder the data structure to ensure a warp coalesces global memory accesses?

    bool *changed = (bool *)malloc(sizeof(bool));
    // cudaMemset((void *)binsDevice, 0, binsSize);

    uint32_t *offsets_device = allocate_and_copy_to_device(offsets, vertex_count + 1);
    uint32_t *indices_device = allocate_and_copy_to_device(indices, edge_count);
    float *weights_device = allocate_and_copy_to_device(weights, edge_count);
    node_data_t *node_data_device = allocate_and_copy_to_device(node_data, vertex_count);
    comm_data_t *comm_data_device = allocate_and_copy_to_device(comm_data, comm_count);
    bool *changed_device = allocate_and_copy_to_device(changed, 1);

    uint32_t *partition = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));
    uint32_t *partition_device = allocate_and_copy_to_device(partition, vertex_count);

    uint32_t *node_part = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));
    uint32_t *node_part_device = allocate_and_copy_to_device(node_part, vertex_count);

    uint32_t *r = (uint32_t *)malloc(vertex_count * sizeof(uint32_t));
    uint32_t *r_device = allocate_and_copy_to_device(r, vertex_count);

    dim3 dim_grid(1);
 	dim3 dim_block(vertex_count);

    // gather_move_candidates_kernel <<<dim_grid, dim_block>>> (offsets_device, indices_device, weights_device, node_data_device, comm_data_device, vertex_count, edge_count, comm_count, gamma);

	move_nodes_fast_kernel <<<dim_grid, dim_block>>> (
        offsets_device,
        indices_device,
        weights_device,
        node_data_device,
        comm_data_device,
        vertex_count,
        edge_count,
        comm_count,
        gamma,
        changed_device,
        partition_device
    );
    cudaDeviceSynchronize();

    copy_from_device(comm_data, comm_data_device, comm_count);

    // TODO: we could use scan kernel here if it is a performance bottleneck, but for now
    // we just do it on cpu
    int partition_count = 0;
    for (int i = 0; i < comm_count; i++) {
        uint32_t agg_count = comm_data[i].agg_count;
        if (agg_count > 0) {
            partition_count++;
        }
    }

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
        vertex_count,
        edge_count,
        gamma,
        partition_device,
        partition_offsets_device, 
        partition_count,
        node_part_device,
        r_len_device,
        r_device
    );
    cudaDeviceSynchronize();

    copy_from_device(r_len, r_len_device, partition_count);

    copy_from_device(partition, partition_device, vertex_count);

    printf("\n---- Partition count %d -----\n\n", partition_count);

    for (int i = 0; i < partition_count; i++) {
        printf("R len for partition %d: %d\n", i, r_len[i]);
    }

    printf("\n\n");

    // for (int i = 0; i < vertex_count; i++) {
    //     printf("Partition idx %d: %d\n", i, partition[i]);
    // }

    for (int i = 0; i < partition_count; i++) {
        uint32_t part_start = partition_offsets[i];
        uint32_t part_end = partition_offsets[i + 1];
        for (int j = part_start; j < part_end; j++) {
            printf("Partition %d: %d\n", i, partition[j]);
        }
    }

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
