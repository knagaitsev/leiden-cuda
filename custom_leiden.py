import time
import random
# import community as community_louvain
import networkx as nx
import copy
import math

class CommunityData:
    def __init__(self, G, community):
        self.G = G
        self.community = community
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
            c = CommunityData(G, community)
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

# TODO: can we maintain a version of P which encompasses P_refined at the same time that we build P_refined?
def merge_nodes_subset(G, p_refined, S: CommunityData, gamma=0.05, theta=1):
    R = []

    remaining_comms = set()

    S_tot = 0
    for node in S.nodes.keys():
        S_tot += vertex_total_edge_weight(G, node)

    for node in S.nodes.keys():
        # consider only nodes that are well connected within subset S, put them in R
        comm = G.nodes[node]["community"]
        remaining_comms.add(comm)

        # these have nothing to do with communities since all nodes will have been placed in singleton communities
        # directly prior to the call of this function -- the only thing we have to be concerned about is self-edges, which should
        # be removed for v_out
        v_in = vertex_total_in_edge_weight(G, node, comm)
        v_tot = vertex_total_edge_weight(G, node)
        v_out = v_tot - v_in

        if v_out >= gamma * v_tot * (S_tot - v_tot):
            R.append(node)

    random.shuffle(R)

    for v in R:
        if not is_in_singleton_community(G, v):
            continue

        # consider only well-connected communities
        T = []
        # this tells us what community in p_refined v belongs to
        curr_comm = G.nodes[v]["community"]
        # this tells us what community in p_new p_refined belongs to
        # p_refined_comm = p_refined.nodes[curr_comm]["community"]
        
        # p_comm_data = p_new.nodes[p_comm]["community_data"]

        for c in remaining_comms:
            p_refined_comm = p_refined.nodes[c]["community"]
            c_in = vertex_total_in_edge_weight(p_refined, c, p_refined_comm)
            c_tot = vertex_total_edge_weight(p_refined, c)
            c_out = c_tot - c_in

            if c_out >= gamma * c_tot * (S_tot - c_tot):
                T.append(c)

        # choose random community C' from T

        probs = []
        for c in T:
            change = move_modularity_change(G, p_refined, v, c)
            if change >= 0:
                probs.append(math.exp((1/theta) * change))
            else:
                probs.append(0)

        print(probs)
        continue

        # TODO: move node v to community C'

        rand_idx = 0
        new_comm = T[rand_idx]
        
        # TODO: need to make sure that this updates the edges and edge weights in p_refined, as
        # this gets used when calculating c_in and c_tot above

        # p_new may also need updating, since it groups p_refined into communities, and we may
        # have just gotten rid of one of the singletons in p_refined
        if new_comm != curr_comm:
            move_node(G, p_refined, v, c)
            remaining_comms.remove(curr_comm)

# returns refined partition
def refine_partition(G, p):
    # TODO: should make a new graph that has G underlying it, without modifying G or p
    # p_refined = p
    # p_refined.graph["parent"] = G

    # set singleton communities to each node
    i = 0
    for node, node_data in G.nodes(data=True):
        node_data["community"] = i
        i += 1

    p_refined = construct_community_graph(G)
    aggregate_graph(G, p_refined)

    # p_new = maintain_p(G, p, p_refined)

    for c, c_data in p.nodes(data=True):
        comm_data = c_data["community_data"]
        merge_nodes_subset(G, p_refined, comm_data)

    return p_refined

# returns new community graph for p_refined, rather than the graph that
# underlies both p and p_refined
# this graph should be the equivalent of P, but now encapsulating P_refined rather than G
# p - the community graph for G
# p_refined - another community graph for G (p_refined will always be the same size or bigger than p)
def maintain_p(G, p, p_refined):
    community_graph = nx.Graph()

    community_graph.graph["parent"] = p_refined

    for p_node, p_node_data in p.nodes(data=True):
        comm_data = p_node_data["community_data"]

        c = CommunityData(p_refined, p_node)

        for node in comm_data.nodes:
            node_data = G.nodes[node]

            p_refined_comm = node_data["community"]

            if not p_refined_comm in c.nodes:
                c.add_node(p_refined_comm)

            p_refined.nodes[p_refined_comm]["community"] = p_node

        community_graph.add_node(p_node, community_data=c)
    
    for u, v, edge_data in p.edges(data=True):
        weight = edge_data.get("weight", 1)

        community_graph.add_edge(u, v, weight=weight)

    # p = construct_community_graph(p_refined)
    # p.graph["parent"] = p_refined

    return community_graph

def assign_singleton_communities(G):
    i = 0

    for node, node_data in G.nodes(data=True):
        node_data["community"] = i

        i += 1

# TODO: do leiden move_nodes_fast
def leiden_move_nodes(G, community_graph):
    while True:
        nodes = list(G.nodes)
        random.shuffle(nodes)

        num_moves = 0

        for node in nodes:
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

    # make sure to clear community_graph edges first
    community_graph.remove_edges_from(list(community_graph.edges))

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



def custom_leiden(G, gamma=1):
    root_graph = G

    num_iter = 0

    assign_singleton_communities(G)

    p = construct_community_graph(G)
    # if there is no parent, it must be the root graph
    p.graph["parent"] = G

    while True:
        print(f"Running Leiden iteration: {num_iter}")
        m = total_edge_weight(G)
        G.graph["m"] = m
        G.graph["gamma"] = gamma

        leiden_move_nodes(G, p)
        if all_communities_one_node(p):
            break

        p_refined = refine_partition(G, p)
        return
        # note -- p_refined is a slightly broken up version of the partitions found in leiden_move_nodes
        aggregate_graph(G, p_refined)
        G = p_refined

        # maintain partition P, this just makes it a partition of p_refined rather than the previous G
        p = maintain_p(G, p, p_refined)

        num_iter += 1

    # since all communities were found to have one node, we can safely propagate from G, rather
    # than from community_graph
    propagate_partitions(G)

    return get_final_communities(root_graph)

def main():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)

    G.add_node(4)
    G.add_node(5)
    G.add_node(6)
    G.add_node(7)

    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 0, weight=1)

    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 3, weight=1)

    G.add_edge(4, 5, weight=1)
    G.add_edge(5, 6, weight=1)
    G.add_edge(6, 7, weight=1)
    G.add_edge(7, 4, weight=1)

    G.add_edge(4, 6, weight=1)
    G.add_edge(5, 7, weight=1)

    G.add_edge(3, 4, weight=1)

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
    print(custom_leiden(G, gamma=0.7))

if __name__ == "__main__":
    main()
