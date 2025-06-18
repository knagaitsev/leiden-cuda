import leidenalg as la

from igraph import Graph
from pathlib import Path
import os

# Get current script directory
curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

# Construct full path to edge list file
data_path = (curr_path / "../validation/clique_ring.txt").resolve()

# Load the graph from the edge list
G = Graph.Read_Edgelist(str(data_path), directed=False)

# G = ig.Graph.Famous('Zachary')

# n_iterations=10
partition = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.05)

print(partition)

# ig.plot(partition)
