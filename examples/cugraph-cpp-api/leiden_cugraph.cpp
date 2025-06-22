#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/graph_functions.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/random/rng_state.hpp>
#include <rmm/device_uvector.hpp>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    raft::handle_t handle;

    using vertex_t = int32_t;
    using edge_t = int32_t;
    using weight_t = float;
    using edge_type_t = int32_t;
    constexpr bool multi_gpu = false;
    // IMPORTANT: must be true if we are doing PageRank
    constexpr bool store_transposed = false;

    // Create the simplest possible graph: just 2 vertices with 1 edge
    std::vector<vertex_t> h_src = {0, 1};
    std::vector<vertex_t> h_dst = {1, 0};

    std::cout << "Created edge list with " << h_src.size() << " edges" << std::endl;

    rmm::device_uvector<vertex_t> d_src(h_src.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_dst(h_dst.size(), handle.get_stream());
    
    raft::update_device(d_src.data(), h_src.data(), h_src.size(), handle.get_stream());
    raft::update_device(d_dst.data(), h_dst.data(), h_dst.size(), handle.get_stream());

    handle.sync_stream();

    cugraph::graph_properties_t graph_properties{
        true, // is_symmetric
        false // is_multigraph
    };

    auto [graph, edge_weights, edge_ids, edge_types, renumber_map] = 
        cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, edge_type_t, store_transposed, multi_gpu>(
            handle,
            std::nullopt, // vertices (let cuGraph determine)
            std::move(d_src),
            std::move(d_dst),
            std::nullopt, // weights
            std::nullopt, // edge_ids
            std::nullopt, // edge_types
            graph_properties,
            false, // renumber
            true // do_expensive_check
        );

    std::cout << "Graph created successfully!" << std::endl;
    std::cout << "Number of vertices: " << graph.number_of_vertices() << std::endl;
    std::cout << "Number of edges: " << graph.number_of_edges() << std::endl;

    auto graph_view = graph.view();
    
    std::cout << "Graph view obtained - vertices: " << graph_view.number_of_vertices() 
                << ", edges: " << graph_view.number_of_edges() << std::endl;
    
    std::cout << "Is symmetric: " << graph_view.is_symmetric() << std::endl;
    std::cout << "Is multigraph: " << graph_view.is_multigraph() << std::endl;
    
    raft::random::RngState rng_state{42};
    
    // auto pagerank_result = raft::make_device_vector<weight_t>(handle, graph_view.number_of_vertices());

    // auto [pageranks, metadata] = cugraph::pagerank<vertex_t, edge_t, weight_t, weight_t, multi_gpu>(
    //     handle,
    //     graph_view,
    //     std::nullopt, // edge_weight_view
    //     std::nullopt, // precomputed_vertex_out_weight_sums
    //     std::nullopt, // personalization
    //     std::nullopt, // initial_pageranks
    //     weight_t{0.85}, // alpha
    //     weight_t{1e-6}, // epsilon
    //     500, // max_iterations
    //     false // do_expensive_check
    // );
    
    // std::vector<weight_t> h_pageranks(pageranks.size());
    // raft::update_host(h_pageranks.data(), pageranks.data(), pageranks.size(), handle.get_stream());
    // handle.sync_stream();

    // for (size_t i = 0; i < h_pageranks.size(); ++i) {
    //     std::cout << "Vertex " << i << " PageRank: " << h_pageranks[i] << std::endl;
    // }
    
    auto [dendrogram, modularity] = cugraph::leiden<vertex_t, edge_t, weight_t, multi_gpu>(
        handle,
        rng_state,
        graph_view,
        std::nullopt, // no edge weights
        1, // max level
        weight_t{1.0}, // resolution
        weight_t{1.0} // theta
    );

    return 0;
}
