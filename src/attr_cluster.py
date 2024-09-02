import networkx as nx
from utils import *

class AttributeHierarchialCluster():
    def __init__(self, cos_grpah:nx.graph) -> None:
        self.graph = cos_graph
        pass

    
    


if __name__ == "__main__":
    base_path = "/home/wangshu/rag/hier_graph_rag/graphrag/ragtest/output/20240813-220313/artifacts"
    
    graph, final_entities, final_relationships = read_graph_nx(base_path)
    cos_graph = compute_distance(graph=graph)

    