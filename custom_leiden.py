import time
import random
# import community as community_louvain
import networkx as nx
import copy
import math
import queue

class CommunityData:
    def __init__(self, G, community):
        self.G = G
        self.community = community
        self.nodes = set()

        self.sum_weights_in = 0
        self.sum_weights_tot = 0

    def add_node(self, node):
        # edges = self.G.edges(node, data=True)
        # for u, v, data in edges:
        #     weight = data.get("weight", 1)

        #     # IMPORTANT: all the sums here are meant to be double counting

        #     self.sum_weights_tot += weight

        #     if u in self.nodes:
        #         self.sum_weights_in += 2 * weight
        #     elif v in self.nodes:
        #         self.sum_weights_in += 2 * weight
        #     elif u == v:
        #         # self-edge gets added here too
        #         self.sum_weights_in += 2 * weight

        self.nodes.add(node)

    def remove_node(self, node):
        self.nodes.remove(node)

        # edges = self.G.edges(node, data=True)
        # for u, v, data in edges:
        #     weight = data.get("weight", 1)

        #     self.sum_weights_tot -= weight

        #     if u in self.nodes:
        #         self.sum_weights_in -= 2 * weight
        #     elif v in self.nodes:
        #         self.sum_weights_in -= 2 * weight

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
        
        # IMPORTANT: this is here because this function should only be used when considering the addition of a node
        # to a community, so we always consider the node self-edge to be a part of that community during the calculation,
        # regardless of the actual current membership of the node
        if node == u and node == v:
            # self-edges should be included here too
            tot += weight
            continue

        # make sure we are not comparing the node's community against it's own community,
        # hence the !=
        if (node != u and community == comm_u) or (node != v and community == comm_v):
            tot += weight

    return tot

def vertex_total_edge_weight(G, node):
    edges = G.edges(node, data=True)

    tot = 0
    for u, v, data in edges:
        weight = data.get("weight", 1)
        tot += weight
    
    return tot

def vertex_candidate_in_edge_weight(G, node, remaining_comms: set):
    edges = G.edges(node, data=True)

    tot = 0
    for u, v, data in edges:
        weight = data.get("weight", 1)
        data_u = G.nodes[u]
        data_v = G.nodes[v]
        comm_u = data_u["community"]
        comm_v = data_v["community"]
        
        # TODO: self-edges should be included here?
        # if node == u and node == v:
        #     tot += weight
        #     continue

        # make sure we are not comparing the node's community against it's own community,
        # hence the !=
        if (node != u and comm_u in remaining_comms) or (node != v and comm_v in remaining_comms):
            tot += weight

    return tot

def comm_candidate_in_edge_weight(G, node, remaining_comms: set):
    edges = G.edges(node, data=True)

    tot = 0
    for u, v, data in edges:
        weight = data.get("weight", 1)
        
        # TODO: self-edges should be included here?
        # if node == u and node == v:
        #     tot += weight
        #     continue

        # make sure we are not comparing the node's community against it's own community,
        # hence the !=
        if (node != u and u in remaining_comms) or (node != v and v in remaining_comms):
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

def cpm(G, gamma):
    tot = 0

    edges = G.edges(data=True)
    for u, v, edge_data in edges:
        weight = edge_data.get("weight", 1)

        data_u = G.nodes[u]
        data_v = G.nodes[v]
        comm_u = data_u["community"]
        comm_v = data_v["community"]

        if comm_u == comm_v:
            tot += weight - gamma

    return tot

def cpm_change(G, community_graph, node, next_comm, gamma):
    curr_comm = G.nodes[node]["community"]
    k_vc_new = vertex_total_in_edge_weight(G, node, next_comm)
    k_vc_old = vertex_total_in_edge_weight(G, node, curr_comm)

    comm_data_old = community_graph.nodes[curr_comm]["community_data"]
    comm_data_new = community_graph.nodes[next_comm]["community_data"]

    delta_H = (k_vc_new - gamma * len(comm_data_new)) - (k_vc_old - gamma * (len(comm_data_old) - 1))

    return delta_H

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

        if u == v:
            continue

        if comm_u == comm_v:
            return False

    return True

# TODO: can we maintain a version of P which encompasses P_refined at the same time that we build P_refined?
def merge_nodes_subset(G, p_refined, S: CommunityData, gamma, theta=1):
    # return early if a subset has only one node
    if len(S.nodes) == 1:
        return

    R = []

    remaining_comms = set()

    for node in S.nodes:
        comm = G.nodes[node]["community"]
        remaining_comms.add(comm)

    S_tot = 0
    for node in S.nodes:
        # S_tot += vertex_total_edge_weight(G, node)

        S_tot += vertex_candidate_in_edge_weight(G, node, remaining_comms)

    print(S_tot)

    # consider only nodes that are well connected within subset S, put them in R
    for node in S.nodes:
        # these have nothing to do with communities since all nodes will have been placed in singleton communities
        # directly prior to the call of this function -- the only thing we have to be concerned about is self-edges, which should
        # be removed for v_out

        # v_in = vertex_total_in_edge_weight(G, node, comm)
        # v_out = v_tot - v_in
        # TODO: should we allow self-edges here? The paper description does not allow self-edges, but it seems
        # they may make sense
        v_in = vertex_candidate_in_edge_weight(G, node, remaining_comms)
        # v_tot = vertex_total_edge_weight(G, node)
        v_tot = v_in

        if v_in >= gamma * v_tot * (S_tot - v_tot):
            R.append(node)

    random.shuffle(R)

    print(f"R: {R}")

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
            # IMPORTANT: it is expected at this point that every p_refined node is in a single community
            # later, they should be joined into the same community
            p_refined_comm = p_refined.nodes[c]["community"]
            # c_in = vertex_total_in_edge_weight(p_refined, c, p_refined_comm)
            # c_out = c_tot - c_in

            # TODO: should we allow self-edges here? The paper description does not allow self-edges, but it seems
            # they may make sense
            c_in = comm_candidate_in_edge_weight(p_refined, c, remaining_comms)
            # c_tot = vertex_total_edge_weight(p_refined, c)
            c_tot = c_in

            if c_in >= gamma * c_tot * (S_tot - c_tot):
                T.append(c)

        # choose random community C' from T

        probs = []
        for c in T:
            # change = move_modularity_change(G, p_refined, v, c)
            change = cpm_change(G, p_refined, v, c, gamma)
            if change >= 0:
                probs.append(math.exp((1/theta) * change))
            else:
                probs.append(0)

        # print(probs)

        idxs = list(range(len(probs)))

        # move node v to community C'

        rand_idx = random.choices(idxs, weights=probs, k=1)[0]
        new_comm = T[rand_idx]
        
        # TODO: need to make sure that this updates the edges and edge weights in p_refined, as
        # this gets used when calculating c_in and c_tot above

        # p_new may also need updating, since it groups p_refined into communities, and we may
        # have just gotten rid of one of the singletons in p_refined
        if new_comm != curr_comm:
            move_node(G, p_refined, v, new_comm)
            remaining_comms.remove(curr_comm)
            add_community_graph_edges_singleton_move(G, p_refined, v)

# returns refined partition
def refine_partition(G, p, gamma):
    # TODO: should make a new graph that has G underlying it, without modifying G or p
    # p_refined = p
    # p_refined.graph["parent"] = G

    # set singleton communities to each node
    i = 0
    for node, node_data in G.nodes(data=True):
        node_data["community"] = i
        i += 1

    p_refined = construct_community_graph(G)
    p_refined.graph["parent"] = G
    aggregate_graph(G, p_refined)

    # p_new = maintain_p(G, p, p_refined)

    for c, c_data in p.nodes(data=True):
        comm_data = c_data["community_data"]
        print(f"Refining a subset of size: {len(comm_data.nodes)}")
        merge_nodes_subset(G, p_refined, comm_data, gamma)

    return p_refined

# important: this must be called before we set G = p_refined, but after we do refining (so G has new community values)
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

# do leiden move_nodes_fast
def move_nodes_fast(G, community_graph, gamma):
    q = queue.Queue()
    in_q = set()

    nodes = list(G.nodes)
    random.shuffle(nodes)

    for node in nodes:
        q.put(node)
        in_q.add(node)
    
    while True:
        if q.empty():
            return
        
        node = q.get()
        in_q.remove(node)

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

            # delta = move_modularity_change(G, community_graph, node, candidate_comm)
            delta = cpm_change(G, community_graph, node, candidate_comm, gamma)

            if delta > best_delta:
                best_delta = delta
                best_comm = candidate_comm
    
        if best_comm != curr_comm:
            # print(f"Best Delta: {best_delta}")

            move_node(G, community_graph, node, best_comm)

            for u, v in G.edges(node):
                data_u = G.nodes[u]
                data_v = G.nodes[v]
                comm_u = data_u["community"]
                comm_v = data_v["community"]

                if comm_u != best_comm and not u in in_q:
                    q.put(u)
                    in_q.add(u)

                if comm_v != best_comm and not v in in_q:
                    q.put(v)
                    in_q.add(v)

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

# this can only be used if a node that was previously in a singleton community gets moved to a new community
def add_community_graph_edges_singleton_move(G, community_graph, node):
    edges = G.edges(node, data=True)
    for u, v, edge_data in edges:
        weight = edge_data.get("weight", 1)

        data_u = G.nodes[u]
        data_v = G.nodes[v]
        comm_u = data_u["community"]
        comm_v = data_v["community"]

        if community_graph.has_edge(comm_u, comm_v):
            edge_data = community_graph[comm_u][comm_v]
            if "weight" in edge_data:
                edge_data["weight"] += weight
            else:
                edge_data["weight"] = weight
        else:
            community_graph.add_edge(comm_u, comm_v, weight=weight)

def propagate_partitions(G, depth=0):
    # we are at the root graph, so nothing to propagate
    if "parent" not in G.graph:
        return
    
    parent = G.graph["parent"]

    for _, comm_node_data in G.nodes(data=True):
        curr_comm = comm_node_data["community"]
        
        comm_data = comm_node_data["community_data"]
        # print(f"PROP: {depth}, comm: {curr_comm}, node count: {len(comm_data.nodes)}")

        for node in comm_data.nodes:
            node_data = parent.nodes[node]
            node_data["community"] = curr_comm
    
    # tail recursion propagating partitions
    propagate_partitions(parent, depth=depth+1)

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
        m = total_edge_weight(G)
        G.graph["m"] = m

        move_nodes_fast(G, p, gamma)
        print(f"Running Leiden iteration: {num_iter}, number of communities after move_nodes_fast: {len(p)}")
        for node, node_data in p.nodes(data=True):
            comm_data = node_data["community_data"]
            # print(f"comm_data len: {len(comm_data)}")

        if all_communities_one_node(p):
            break

        # for node, node_data in p.nodes(data=True):
        #     comm_data = node_data["community_data"]
        #     print(f"Community: {node}, Nodes len: {len(comm_data.nodes)}")

        # for node, node_data in G.nodes(data=True):
        #     comm = node_data["community"]
        #     print(f"Node community: {comm}")

        p_refined = refine_partition(G, p, gamma)
        
        # maintain partition P, this just makes it a partition of p_refined rather than the previous G
        # IMPORTANT: this must come before G = p_refined
        p = maintain_p(G, p, p_refined)

        G = p_refined

        num_iter += 1

        # if num_iter > 10:
        #     return

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

    # assign_singleton_communities(G)
    # community_graph = construct_community_graph(G)

    gamma = 0.09

    # print(f"Start CPM: {cpm(G, gamma)}")

    # # delta = cpm_change(G, community_graph, 0, 1, gamma)
    # # print(f"Delta: {delta}")
    # # move_node(G, community_graph, 0, 1)

    # move_nodes_fast(G, community_graph, gamma)

    # print(f"End CPM: {cpm(G, gamma)}")

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
    print(custom_leiden(G, gamma))

if __name__ == "__main__":
    main()
