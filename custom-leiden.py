import time
# import community as community_louvain
import networkx as nx

class CommunityData:
    def __init__(self, G):
        self.G = G
        self.nodes = {}

        self.sum_weights_in = 0
        self.sum_weights_tot = 0

    def add_node(self, node):
        edges = self.G.edges(node, data=True)
        for u, v, data in edges:
            weight = data.get("weight", 1)

            # If u or v are in self.nodes, then we do not need to increment self.sum_weights_tot
            # as it will already be in there, given that this node was adjacent to the community.
            # We do not want to double count it.

            if u in self.nodes:
                self.sum_weights_in += weight
            elif v in self.nodes:
                self.sum_weights_in += weight
            else:
                self.sum_weights_tot += weight
        
        self.nodes[node] = True

    def nodes(self):
        return self.nodes.keys()

def total_edge_weight(G):
    tot = 0

    for u, v, data in G.edges(data=True):
        tot += data.get("weight", 1)

    return tot

# important: The "or" in here makes the assumption that this vertex is not yet
# a part of the community (it would need to be "and" if already a member)
def vertex_total_in_edge_weight(G, node, community):
    edges = G.edges(node, data=True)

    tot = 0
    for u, v, data in edges:
        weight = data.get("weight", 1)
        data_u = G.nodes[u]
        data_v = G.nodes[v]
        comm_u = data_u["community"]
        comm_v = data_v["community"]

        if community == comm_u or community == comm_v:
            tot += weight
    
    return tot

def vertex_total_edge_weight(G, node):
    edges = G.edges(node, data=True)

    tot = 0
    for u, v, data in edges:
        weight = data.get("weight", 1)
        tot += weight
    
    return tot

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
    # TODO: we should be passing this in, not recomputing it
    m = total_edge_weight(G)

    comm_data = community_graph.nodes[community]["community_data"]

    sum_in = comm_data.sum_weights_in
    sum_tot = comm_data.sum_weights_tot

    print(sum_in, sum_tot)

    # TODO: could be more efficient by counting these both at once
    k_i_in = vertex_total_in_edge_weight(G, node, community)
    k_i = vertex_total_edge_weight(G, node)

    print(k_i_in, k_i)

    new_comm_Q = ((sum_in + k_i_in) / (2 * m)) - ((sum_tot + k_i) / (2 * m))**2
    prev_comm_Q = (sum_in / (2 * m)) - (sum_tot / (2 * m))**2 - (k_i / (2 * m))**2

    print(new_comm_Q, prev_comm_Q)

    delta_Q = new_comm_Q - prev_comm_Q

    return delta_Q

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

    c0 = CommunityData(G)
    c0.add_node(0)

    c1 = CommunityData(G)
    c1.add_node(1)
    c1.add_node(2)
    c1.add_node(3)

    community_graph.add_node(0, community_data=c0)
    community_graph.add_node(1, community_data=c1)

    print(modularity(G))

    print(isolated_move_modularity_change(G, community_graph, 0, 1))

main()
