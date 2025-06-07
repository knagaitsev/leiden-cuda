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

def modularity(G):
    # TODO: get total edge weight
    m = 4

    tot = 0

    for i, data_i in G.nodes(data=True):
        for j, data_j in G.nodes(data=True):
            same_comm = data_i["community"] == data_j["community"]
            if not same_comm:
                continue
            
            # TODO
    
    Q = (1 / (2 * m))

    return Q

def leiden(G):
    pass

def main():
    G = nx.Graph()
    G.add_node(0, community=0)
    G.add_node(1, community=0)
    G.add_node(2, community=0)
    G.add_node(3, community=0)

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 0)

    # G = nx.read_edgelist("../validation/clique_ring.txt", nodetype=int)

    community_graph = nx.Graph()

    p1 = CommunityData()

    p1.add_node(0)
    p1.add_node(1)
    p1.add_node(2)
    p1.add_node(3)

    community_graph.add_node(0, community_data=p1)

    print(modularity(G))

main()
