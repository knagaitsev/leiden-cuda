import time
# import community as community_louvain
import networkx as nx
from networkx.algorithms.community import louvain_communities, leiden_communities

# Create a graph
G = nx.karate_club_graph()
# G = nx.read_edgelist("../data/flickr-groupmemberships/out.flickr-groupmemberships", comments="%")
print(G)

# Compute communities using the Louvain method
start = time.time()

# communities = leiden_communities(G, seed=42, resolution=1, backend="cugraph")
communities = louvain_communities(G, seed=42, resolution=1, backend="cugraph")

# communities = louvain_communities(G, seed=42, resolution=1, backend="parallel")
end = time.time()

print(f"Runtime: {end - start:.4f} seconds")

print(f"Community count: {len(communities)}")
# Print the communities
# for i, community in enumerate(communities):
#     print(f"Community {i+1}: {sorted(community)}")
