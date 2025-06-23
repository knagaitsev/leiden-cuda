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
__global__ void gather_move_candidates_kernel(uint32_t *offsets, uint32_t *indices, float *weights, node_data_t *node_data, comm_data_t *comm_data, int vertex_count, int edge_count, int comm_count, float gamma) {
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
    }
}

// two approaches to doing move_nodes_fast: parallelizing at node level is below
// - another option is parallelizing at edge level, letting each thread consider an edge
__global__ void move_nodes_fast_kernel(uint32_t *offsets, uint32_t *indices, float *weights, node_data_t *node_data, comm_data_t *comm_data, int vertex_count, int edge_count, int comm_count, float gamma) {
    
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
T* copy_from_device(T* data_host, T* data_device, int len) {
    int size = len * sizeof(T);
    cudaMemcpy(data_host, data_device, size, cudaMemcpyDeviceToHost);
}

extern "C" void move_nodes_fast(uint32_t *offsets, uint32_t *indices, float *weights, node_data_t *node_data, comm_data_t *comm_data, int vertex_count, int edge_count, int comm_count, float gamma) {
    // each thread of the cuda kernel considers one node and attempts to greedily increase the CPM
    // by moving it to the best neighboring community

    // the threads should be considering nodes in a semi-random order though.
    // will it be better to give them an array of random indices to access,
    // - or should we reorder the data structure to ensure a warp coalesces global memory accesses?

    uint32_t *offsets_device = allocate_and_copy_to_device(offsets, vertex_count + 1);
    uint32_t *indices_device = allocate_and_copy_to_device(indices, edge_count);
    float *weights_device = allocate_and_copy_to_device(weights, edge_count);
    node_data_t *node_data_device = allocate_and_copy_to_device(node_data, vertex_count);
    comm_data_t *comm_data_device = allocate_and_copy_to_device(comm_data, comm_count);

    dim3 dim_grid(1);
 	dim3 dim_block(vertex_count);

    gather_move_candidates_kernel <<<dim_grid, dim_block>>> (offsets_device, indices_device, weights_device, node_data_device, comm_data_device, vertex_count, edge_count, comm_count, gamma);

	// move_nodes_fast_kernel <<<dim_grid, dim_block>>> (offsets_device, indices_device, weights_device, node_data_device, comm_data_device, vertex_count, edge_count, comm_count, gamma);

    cudaDeviceSynchronize();

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
