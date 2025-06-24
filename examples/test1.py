import time
# import community as community_louvain
import networkx as nx
from networkx.algorithms.community import louvain_communities, leiden_communities
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent

if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from custom_louvain import custom_louvain
from custom_leiden import custom_leiden

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

# Create a graph
# G = nx.karate_club_graph()
# G = nx.read_edgelist(curr_path / "../data/flickr-groupmemberships/out.flickr-groupmemberships", comments="%")
G = nx.read_edgelist(curr_path / "../data/arenas-jazz/out.arenas-jazz", comments="%")

# data_path = Path.resolve(curr_path / "../validation/clique_ring.txt")
# G = nx.read_edgelist(data_path, nodetype=int)

# data_path = Path.resolve(curr_path / "../validation/clique_ring_weighted.txt")
# G = nx.read_weighted_edgelist(data_path, nodetype=int)

top_10 = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]

# Print the top 10
for node, degree in top_10:
    print(f"Node {node} has degree {degree}")

print("Number of vertices:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

has_self_loops = any(G.has_edge(n, n) for n in G.nodes())

print(f"Has self loops: {has_self_loops}")

# Compute communities using the Louvain method
start = time.time()

communities = leiden_communities(G, seed=42, resolution=1, backend="cugraph")
# communities = louvain_communities(G, seed=42, resolution=1, backend="cugraph")
# communities = custom_louvain(G)
# communities = custom_leiden(G, gamma=0.05, max_iter=0)

# communities = louvain_communities(G, seed=42, resolution=1, backend="parallel")
end = time.time()

print(f"Runtime: {end - start:.4f} seconds")

print(f"Community count: {len(communities)}")

exit(0)

# Print the communities
for i, community in enumerate(communities):
    print(f"Community {i+1}: {sorted(community)}")

node_color_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_color_map[node] = i

# Generate a color for each community
colors = [node_color_map[node] for node in G.nodes()]
# print(colors)

# Draw the graph
pos = nx.spring_layout(G, seed=42)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set3, node_size=500)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# plt.show()

figs_dir = Path.resolve(curr_path / "../figs")

# plt.savefig(figs_dir / "jazz_leiden.pdf")
plt.savefig(figs_dir / "clique_ring_leiden.pdf")
