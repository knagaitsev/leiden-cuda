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

            # IMPORTANT: all the sums here are meant to be double counting

            self.sum_weights_tot += weight

            if u in self.nodes:
                self.sum_weights_in += 2 * weight
            elif v in self.nodes:
                self.sum_weights_in += 2 * weight
        
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
    m = G.graph["m"]

    tot = 0

    for i, data_i in G.nodes(data=True):
        for j, data_j in G.nodes(data=True):
            same_comm = data_i["community"] == data_j["community"]
            if not same_comm:
                continue

            if i == j:
                k_i = vertex_total_edge_weight(G, i)

                v = - ((k_i**2) / (2 * m))

                tot += v
            elif G.has_edge(i, j):

                data_ij = G[i][j]
                A_ij = data_ij.get("weight", 1)
                
                k_i = vertex_total_edge_weight(G, i)
                k_j = vertex_total_edge_weight(G, j)

                v = A_ij - ((k_i * k_j) / (2 * m))

                tot += v
    
    Q = (1 / (2 * m)) * tot

    return Q

def isolated_move_modularity_change(G, community_graph, node, community):
    m = G.graph["m"]

    comm_data = community_graph.nodes[community]["community_data"]

    sum_in = comm_data.sum_weights_in
    sum_tot = comm_data.sum_weights_tot

    # print(sum_in, sum_tot)

    # TODO: could be more efficient by counting these both at once
    k_i_in = vertex_total_in_edge_weight(G, node, community)
    k_i = vertex_total_edge_weight(G, node)

    # print(k_i_in, k_i)

    new_comm_Q = ((sum_in + 2 * k_i_in) / (2 * m)) - ((sum_tot + k_i) / (2 * m))**2
    prev_comm_Q = (sum_in / (2 * m)) - (sum_tot / (2 * m))**2 - (k_i / (2 * m))**2

    # print(new_comm_Q, prev_comm_Q)

    delta_Q = new_comm_Q - prev_comm_Q

    return delta_Q

def move_isolated_node(G: nx.Graph, community_graph: nx.Graph, node, community):
    node_data = G.nodes[node]
    prev_community = node_data["community"]

    community_graph.remove_node(prev_community)

    node_data["community"] = community

    new_community = community_graph.nodes[community]["community_data"]
    new_community.add_node(node)

def construct_community_graph(G: nx.Graph):
    community_graph = nx.Graph()

    for node, node_data in G.nodes(data=True):
        community = node_data["community"]

        if community in community_graph:
            c = community_graph.nodes[community]["community_data"]
            c.add_node(node)
        else:
            c = CommunityData(G)
            c.add_node(node)

            community_graph.add_node(community, community_data=c)

        edges = G.edges(node, data=True)
        for u, v, edge_data in edges:
            weight = edge_data.get("weight", 1)
            
            data_u = G.nodes[u]
            data_v = G.nodes[v]
            comm_u = data_u["community"]
            comm_v = data_v["community"]

            if comm_u != comm_v and comm_u in community_graph and comm_v in community_graph:
                if (comm_u, comm_v) in community_graph.edges:
                    edge_data = community_graph.get_edge_data(comm_u, comm_v)
                    edge_data["weight"] += weight
                else:
                    community_graph.add_edge(comm_u, comm_v, weight=weight)

    return community_graph

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

    m = total_edge_weight(G)
    G.graph["m"] = m
    print(f"Total edge weight: {m}")

    # G = nx.read_edgelist("../validation/clique_ring.txt", nodetype=int)

    # community_graph = nx.Graph()
    # c0 = CommunityData(G)
    # c0.add_node(0)
    # c1 = CommunityData(G)
    # c1.add_node(1)
    # c1.add_node(2)
    # c1.add_node(3)
    # community_graph.add_node(0, community_data=c0)
    # community_graph.add_node(1, community_data=c1)

    community_graph = construct_community_graph(G)

    print(f"Modularity: {modularity(G)}")

    print(f"Delta: {isolated_move_modularity_change(G, community_graph, 0, 1)}")

    move_isolated_node(G, community_graph, 0, 1)

    print(f"Modularity: {modularity(G)}")

main()
