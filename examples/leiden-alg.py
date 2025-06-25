import leidenalg as la

from igraph import Graph
from pathlib import Path
import os

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

data_path = (curr_path / "../data/arenas-jazz/out.arenas-jazz").resolve()
# data_path = (curr_path / "../data/flickr-groupmemberships/out.flickr-groupmemberships").resolve()
edges = []
with open(data_path, 'r') as f:
    for line in f:
        if not line.startswith('%'):
            u, v = map(int, line.strip().split())
            edges.append((u, v))
G = Graph(edges=edges, directed=False)

# data_path = (curr_path / "../validation/clique_ring.txt").resolve()
# G = Graph.Read_Edgelist(str(data_path), directed=False)

# G = ig.Graph.Famous('Zachary')

# n_iterations=10
# partition = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.05)
# print(partition)

optimiser = la.Optimiser()
partition = la.CPMVertexPartition(G, resolution_parameter=0.05)

# res = optimiser.move_nodes(partition)
# print(res)

diff = optimiser.optimise_partition(partition, n_iterations=-1)
print(diff)

# ig.plot(partition)
