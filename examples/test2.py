import time
import os
from pathlib import Path

import cudf
import cugraph

import networkx as nx
import pandas as pd
from scipy.sparse import coo_matrix

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

# G_nx = nx.karate_club_graph()
# G_nx = nx.read_edgelist(curr_path / "../data/flickr-groupmemberships/out.flickr-groupmemberships", comments="%")
G_nx = nx.read_edgelist(curr_path / "../data/wikipedia_link_mi/out.wikipedia_link_mi", comments="%")

coo = nx.to_scipy_sparse_array(G_nx, format='coo')

csr = coo.tocsr()

offsets = pd.Series(csr.indptr.astype('int32'))
indices = pd.Series(csr.indices.astype('int32'))

print(f"Offsets len: {len(offsets)}")
print(f"Indices len: {len(indices)}")

G = cugraph.from_adjlist(offsets, indices, None)

start = time.time()
partitions = cugraph.leiden(G)
end = time.time()

print(f"Runtime: {end - start:.4f} seconds")

# edges = cudf.DataFrame({
#     'src': [0, 1, 2, 3, 4, 5, 6, 7],
#     'dst': [1, 2, 0, 4, 5, 6, 7, 4],
#     'weight': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
# })

# G = cugraph.Graph()
# G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='weight')

# partitions = cugraph.leiden(G)

# print(partitions)
