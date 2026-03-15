import networkx as nx
from pyvis.network import Network

def build_knowledge_graph(entities, relations):
    """Build NetworkX graph from entities and relations"""
    G = nx.Graph()
    
    # Add nodes
    for entity in entities:
        G.add_node(entity.id, label=entity.name, type=entity.type)
    
    # Add edges
    for relation in relations:
        G.add_edge(relation.entity1_id, relation.entity2_id, 
                  label=relation.relation_type, 
                  confidence=relation.confidence)
    
    return G

def get_subgraph(graph, center_entity, depth=2):
    """Get subgraph centered around an entity"""
    if center_entity not in graph:
        return nx.Graph()
    
    nodes = set([center_entity])
    current_nodes = set([center_entity])
    
    for _ in range(depth):
        next_nodes = set()
        for node in current_nodes:
            neighbors = set(graph.neighbors(node))
            next_nodes.update(neighbors)
        nodes.update(next_nodes)
        current_nodes = next_nodes
    
    return graph.subgraph(nodes)