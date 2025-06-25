import time
import os
from pathlib import Path

import cudf
import cugraph

import networkx as nx
import pandas as pd
from scipy.sparse import coo_matrix
import json

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

# G_nx = nx.karate_club_graph()
# G_nx = nx.read_edgelist(curr_path / "../data/flickr-groupmemberships/out.flickr-groupmemberships", comments="%")
# G_nx = nx.read_edgelist(curr_path / "../data/wikipedia_link_mi/out.wikipedia_link_mi", comments="%")
# G_nx = nx.read_edgelist(curr_path / "../data/dimacs10-uk-2002/out.dimacs10-uk-2002", comments="%")
G_nx = nx.read_edgelist(curr_path / "../data/youtube-links/out.youtube-links", comments="%")
# G_nx = nx.read_edgelist(curr_path / "../data/arenas-jazz/out.arenas-jazz", comments="%")

top_10 = sorted(G_nx.degree, key=lambda x: x[1], reverse=True)[:10]

# Print the top 10
for node, degree in top_10:
    print(f"Node {node} has degree {degree}")

num_nodes = G_nx.number_of_nodes()
num_edges = G_nx.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

coo = nx.to_scipy_sparse_array(G_nx, format='coo')

csr = coo.tocsr()

offsets = pd.Series(csr.indptr.astype('int32'))
indices = pd.Series(csr.indices.astype('int32'))

print(f"Offsets len: {len(offsets)}")
print(f"Indices len: {len(indices)}")

G = cugraph.from_adjlist(offsets, indices, None)

# warmup
cugraph.leiden(G)

runtimes = []

for i in range(1, 11):
    start = time.time()
    partitions = cugraph.leiden(G, max_iter=i)
    end = time.time()
    runtime = end - start
    print(f"Runtime: {end - start:.4f} seconds, max_iter={i}")

    runtimes.append({
        "max_iter": i,
        "runtime": runtime
    })

with open(curr_path / "../results/cugraph_vary_max_iter.json", "w") as f:
    json.dump(runtimes, f)

# edges = cudf.DataFrame({
#     'src': [0, 1, 2, 3, 4, 5, 6, 7],
#     'dst': [1, 2, 0, 4, 5, 6, 7, 4],
#     'weight': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
# })

# G = cugraph.Graph()
# G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='weight')

# partitions = cugraph.leiden(G)

# print(partitions)
