#include <cuda_runtime.h>
#include <iostream>

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

__global__ void move_nodes_fast_kernel(uint32_t *offsets, uint32_t *indices, float *weights, uint32_t *communities, int vertex_count, int edge_count) {
    communities[threadIdx.x] = 1;
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

extern "C" void move_nodes_fast(uint32_t *offsets, uint32_t *indices, float *weights, uint32_t *communities, int vertex_count, int edge_count) {
    // each thread of the cuda kernel considers one node and attempts to greedily increase the CPM
    // by moving it to the best neighboring community

    // the threads should be considering nodes in a semi-random order though.
    // will it be better to give them an array of random indices to access,
    // - or should we reorder the data structure to ensure a warp coalesces global memory accesses?

    uint32_t *offsets_device = allocate_and_copy_to_device(offsets, vertex_count + 1);
    uint32_t *indices_device = allocate_and_copy_to_device(indices, edge_count);
    float *weights_device = allocate_and_copy_to_device(weights, edge_count);
    uint32_t *communities_device = allocate_and_copy_to_device(communities, vertex_count);

    dim3 dim_grid(1);
 	dim3 dim_block(vertex_count);

	move_nodes_fast_kernel <<<dim_grid, dim_block>>> (offsets_device, indices_device, weights_device, communities_device, vertex_count, edge_count);

    cudaDeviceSynchronize();

    copy_from_device(offsets, offsets_device, vertex_count + 1);
    copy_from_device(indices, indices_device, edge_count);
    copy_from_device(weights, weights_device, edge_count);
    copy_from_device(communities, communities_device, vertex_count);

    cudaFree(offsets_device);
    cudaFree(indices_device);
    cudaFree(weights_device);
    cudaFree(communities_device);
}
