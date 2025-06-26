import leidenalg as la

from igraph import Graph
from pathlib import Path
import os
import time

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

# multiplied by 2 at end, as this seems to be the norm for
# the leidenalg package and other CPM computations
def compute_cpm(partition, gamma):
    total = 0.0
    for comm in partition:
        n_c = len(comm)
        internal_weight = 0.0
        for i in comm:
            neighbors = partition.graph.neighbors(i)
            for j in neighbors:
                if j in comm and j > i:  # avoid double-counting
                    internal_weight += 1  # edge weight is 1
        total += internal_weight - gamma * (n_c * (n_c - 1) / 2)
    return total * 2

data_path = (curr_path / "../data/wikipedia_link_mi/out.wikipedia_link_mi").resolve()
# data_path = (curr_path / "../data/youtube-links/out.youtube-links").resolve()
# data_path = (curr_path / "../data/arenas-jazz/out.arenas-jazz").resolve()
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

gamma = 0.05
n_iter=1


# optimiser = la.Optimiser()
# partition = la.CPMVertexPartition(G, resolution_parameter=0.05)
# start = time.time()
# # partition = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=gamma, n_iterations=n_iter)
# res = optimiser.move_nodes(partition)
# end = time.time()
# runtime = end - start
# print(f"Runtime: {end - start:.4f} seconds")

# print(partition)

# cpm_value = compute_cpm(partition, gamma)
# print("CPM value:", cpm_value)

# optimiser = la.Optimiser()
# partition = la.CPMVertexPartition(G, resolution_parameter=0.05)
# res = optimiser.move_nodes(partition)
# print(res)

optimiser = la.Optimiser()
partition = la.CPMVertexPartition(G, resolution_parameter=0.05)
diff = optimiser.optimise_partition(partition, n_iterations=1)
print(diff)

# ig.plot(partition)
