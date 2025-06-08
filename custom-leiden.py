import time
# import community as community_louvain
import networkx as nx

class CommunityData:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_id):
        self.nodes[node_id] = True

    def nodes(self):
        return self.nodes.keys()

def total_edge_weight(G):
    tot = 0

    for u, v, data in G.edges(data=True):
        tot += data.get("weight", 1)

    return tot

def vertex_total_edge_weight(G, node):
    edges = G.edges(node, data=True)

    tot = 0
    for u, v, data in edges:
        weight = data.get("weight", 1)
        tot += weight
    
    return weight

def modularity(G):
    m = total_edge_weight(G)
    print(f"Total edge weight: {m}")

    tot = 0

    for i, data_i in G.nodes(data=True):
        for j, data_j in G.nodes(data=True):
            same_comm = data_i["community"] == data_j["community"]
            if not same_comm:
                continue

            if not G.has_edge(i, j):
                continue

            data_ij = G[i][j]
            A_ij = data_ij.get("weight", 1)
            
            k_i = vertex_total_edge_weight(G, i)
            k_j = vertex_total_edge_weight(G, j)

            v = A_ij - ((k_i * k_j) / (2 * m))

            tot += v
    
    Q = (1 / (2 * m)) * tot

    return Q

def isolated_move_modularity_change(G, community_graph, node, community):
    return 0

def leiden(G):
    pass

def main():
    G = nx.Graph()
    G.add_node(0, community=0)
    G.add_node(1, community=1)
    G.add_node(2, community=1)
    G.add_node(3, community=1)

    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 0, weight=1)

    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 3, weight=1)

    # G = nx.read_edgelist("../validation/clique_ring.txt", nodetype=int)

    community_graph = nx.Graph()

    c0 = CommunityData()
    c0.add_node(0)

    c1 = CommunityData()
    c1.add_node(1)
    c1.add_node(2)
    c1.add_node(3)

    community_graph.add_node(0, community_data=c0)
    community_graph.add_node(1, community_data=c1)

    print(modularity(G))

    print(isolated_move_modularity_change(G, community_graph, 0, 1))

main()
