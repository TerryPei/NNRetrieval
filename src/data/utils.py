from graphviz import Digraph


def visualize_graph(edges_list, filename):

    unique_nodes = []
    # print(network.edges_list)
    for node_i, node_j in edges_list:
        unique_nodes.append(node_i)
        unique_nodes.append(node_j)
    unique_nodes = set(unique_nodes)

    graph_dot = Digraph('round-table', node_attr=dict(style='filled'))
    for node in unique_nodes:
        if 'CONCAT' in node:
            graph_dot.node(node, shape='box', fillcolor='palegoldenrod')
        elif 'ADD' in node:
            graph_dot.node(node, shape='diamond', fillcolor='lightblue1')
        elif 'stem.' in node:
            graph_dot.node(node, shape='box', fillcolor='darkseagreen1')
        elif 'head.' in node:
            graph_dot.node(node, shape='box', fillcolor='darkseagreen1')
        else:
            graph_dot.node(node, fillcolor='gray85')
    graph_dot.edges(edges_list)

    graph_dot.render(filename=filename)
