import time
import random
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
            elif u == v:
                # self-edge gets added here too
                self.sum_weights_in += 2 * weight

        self.nodes[node] = True

    def remove_node(self, node):
        del self.nodes[node]

        edges = self.G.edges(node, data=True)
        for u, v, data in edges:
            weight = data.get("weight", 1)

            self.sum_weights_tot -= weight

            if u in self.nodes:
                self.sum_weights_in -= 2 * weight
            elif v in self.nodes:
                self.sum_weights_in -= 2 * weight

    def nodes(self):
        return self.nodes.keys()

    def __len__(self):
        return len(self.nodes)

def total_edge_weight(G):
    tot = 0

    for u, v, data in G.edges(data=True):
        tot += data.get("weight", 1)

    return tot

def vertex_total_in_edge_weight(G, node, community):
    edges = G.edges(node, data=True)

    tot = 0
    for u, v, data in edges:
        weight = data.get("weight", 1)
        data_u = G.nodes[u]
        data_v = G.nodes[v]
        comm_u = data_u["community"]
        comm_v = data_v["community"]
        
        # make sure we are not comparing the node's community against it's own community,
        # hence the !=
        if (node != u and community == comm_u) or (node != v and community == comm_v):
            tot += weight
        elif node == u and node == v:
            # self-edges should be included here too
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

            if G.has_edge(i, j):
                data_ij = G[i][j]
                A_ij = data_ij.get("weight", 1)
                
                k_i = vertex_total_edge_weight(G, i)
                k_j = vertex_total_edge_weight(G, j)

                v = A_ij - ((k_i * k_j) / (2 * m))

                tot += v
            elif i == j:
                k_i = vertex_total_edge_weight(G, i)

                v = - ((k_i**2) / (2 * m))

                tot += v

    Q = (1 / (2 * m)) * tot

    return Q

def get_comm_Q(m, community_graph, comm, diff_k_i_in, diff_k_i):
    comm_data = community_graph.nodes[comm]["community_data"]

    sum_in = comm_data.sum_weights_in
    sum_tot = comm_data.sum_weights_tot

    return ((sum_in + 2 * diff_k_i_in) / (2 * m)) - ((sum_tot + diff_k_i) / (2 * m))**2


def move_modularity_change(G, community_graph, node, next_comm):
    m = G.graph["m"]

    node_data = G.nodes[node]
    curr_comm = node_data["community"]

    # TODO: could be more efficient by counting these both at once
    curr_k_i_in = vertex_total_in_edge_weight(G, node, curr_comm)
    next_k_i_in = vertex_total_in_edge_weight(G, node, next_comm)
    k_i = vertex_total_edge_weight(G, node)

    # new_comm_Q = ((sum_in + 2 * k_i_in) / (2 * m)) - ((sum_tot + k_i) / (2 * m))**2
    # old_comm_Q = (sum_in / (2 * m)) - (sum_tot / (2 * m))**2 - (k_i / (2 * m))**2

    curr_old_Q = get_comm_Q(m, community_graph, curr_comm, 0, 0)
    curr_new_Q = get_comm_Q(m, community_graph, curr_comm, -curr_k_i_in, -k_i)

    next_old_Q = get_comm_Q(m, community_graph, next_comm, 0, 0)
    next_new_Q = get_comm_Q(m, community_graph, next_comm, next_k_i_in, k_i)

    new_Q = next_new_Q + curr_new_Q
    old_Q = next_old_Q + curr_old_Q

    # print(new_comm_Q, prev_comm_Q)

    delta_Q = new_Q - old_Q

    return delta_Q

def move_node(G: nx.Graph, community_graph: nx.Graph, node, community):
    node_data = G.nodes[node]
    prev_community = node_data["community"]

    prev_comm_data = community_graph.nodes[prev_community]["community_data"]
    prev_comm_data.remove_node(node)

    if len(prev_comm_data) == 0:
        community_graph.remove_node(prev_community)

    node_data["community"] = community

    new_comm_data = community_graph.nodes[community]["community_data"]
    new_comm_data.add_node(node)

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

        # edges = G.edges(node, data=True)
        # for u, v, edge_data in edges:
        #     weight = edge_data.get("weight", 1)
            
        #     data_u = G.nodes[u]
        #     data_v = G.nodes[v]
        #     comm_u = data_u["community"]
        #     comm_v = data_v["community"]

        #     if comm_u != comm_v and comm_u in community_graph and comm_v in community_graph:
        #         if (comm_u, comm_v) in community_graph.edges:
        #             edge_data = community_graph.get_edge_data(comm_u, comm_v)
        #             edge_data["weight"] += weight
        #         else:
        #             community_graph.add_edge(comm_u, comm_v, weight=weight)

    assign_singleton_communities(community_graph)

    return community_graph

def assign_singleton_communities(G):
    i = 0

    for node, node_data in G.nodes(data=True):
        node_data["community"] = i

        i += 1

def is_in_singleton_community(G, node):
    edges = G.edges(node, data=True)

    for u, v, edge_data in edges:
        data_u = G.nodes[u]
        data_v = G.nodes[v]
        comm_u = data_u["community"]
        comm_v = data_v["community"]

        if comm_u == comm_v:
            return False

    return True

def louvain_move_nodes(G, community_graph):
    while True:
        nodes = list(G.nodes)
        random.shuffle(nodes)

        num_moves = 0

        for node in nodes:
            # if not is_in_singleton_community(G, node):
            #     continue

            node_data = G.nodes[node]
            curr_comm = node_data["community"]

            best_comm = curr_comm
            best_delta = 0
            edges = G.edges(node, data=True)
            for u, v, edge_data in edges:
                data_u = G.nodes[u]
                data_v = G.nodes[v]
                comm_u = data_u["community"]
                comm_v = data_v["community"]

                candidate_comm = comm_u
                if comm_u == curr_comm:
                    candidate_comm = comm_v
                
                if candidate_comm == curr_comm or candidate_comm == best_comm:
                    continue

                delta = move_modularity_change(G, community_graph, node, candidate_comm)

                if delta > best_delta:
                    best_delta = delta
                    best_comm = candidate_comm
        
            if best_comm != curr_comm:
                # print(f"Delta: {best_delta}")
                # print(f"MOVED {node} {best_comm}")
                move_node(G, community_graph, node, best_comm)

                num_moves += 1

        if num_moves == 0:
            break

def all_communities_one_node(community_graph):
    for _, node_data in community_graph.nodes(data=True):
        comm_data = node_data["community_data"]
        if len(comm_data) == 0:
            raise Exception("each community should have at least one node")
        elif len(comm_data) > 1:
            return False
    
    return True

def aggregate_graph(G, community_graph):
    # for comm, comm_node_data in community_graph.nodes(data=True):
    #     comm_data = comm_node_data["community_data"]

    edges = G.edges(data=True)
    for u, v, edge_data in edges:
        weight = edge_data.get("weight", 1)

        data_u = G.nodes[u]
        data_v = G.nodes[v]
        comm_u = data_u["community"]
        comm_v = data_v["community"]

        # add edge weight to the community_graph, simply increasing the weight if the edge already exists
        # Important: this should include adding self-edges for the community to itself

        if community_graph.has_edge(comm_u, comm_v):
            edge_data = community_graph[comm_u][comm_v]
            if "weight" in edge_data:
                edge_data["weight"] += weight
            else:
                edge_data["weight"] = weight
        else:
            community_graph.add_edge(comm_u, comm_v, weight=weight)

def propagate_partitions(G):
    # we are at the root graph, so nothing to propagate
    if "parent" not in G.graph:
        return
    
    parent = G.graph["parent"]

    for _, comm_node_data in G.nodes(data=True):
        curr_comm = comm_node_data["community"]
        
        comm_data = comm_node_data["community_data"]
        for node in comm_data.nodes.keys():
            node_data = parent.nodes[node]
            node_data["community"] = curr_comm
    
    # tail recursion propagating partitions
    propagate_partitions(parent)

def get_final_communities(G):
    comms = {}

    for node, data in G.nodes(data=True):
        comm = data["community"]
        if comm in comms:
            comms[comm].append(node)
        else:
            comms[comm] = [node]

    return list(comms.values())

def custom_louvain(G):
    root_graph = G

    num_iter = 0

    assign_singleton_communities(G)

    while True:
        print(f"Running Louvain iteration: {num_iter}")
        m = total_edge_weight(G)
        G.graph["m"] = m

        community_graph = construct_community_graph(G)
        # if there is no parent, it must be the root graph
        community_graph.graph["parent"] = G

        louvain_move_nodes(G, community_graph)
        if all_communities_one_node(community_graph):
            break

        aggregate_graph(G, community_graph)
        G = community_graph

        num_iter += 1

    # since all communities were found to have one node, we can safely propagate from G, rather
    # than from community_graph
    propagate_partitions(G)

    return get_final_communities(root_graph)

def main():
    G = nx.Graph()
    G.add_node(0, community=0)
    G.add_node(1, community=1)
    G.add_node(2, community=1)
    G.add_node(3, community=1)

    assign_singleton_communities(G)

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

    # community_graph = construct_community_graph(G)
    # print(f"Modularity: {modularity(G)}")
    # louvain_move_nodes(G, community_graph)
    # print(f"Modularity: {modularity(G)}")
    print(custom_louvain(G))

if __name__ == "__main__":
    main()
