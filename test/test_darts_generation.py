import numpy as np

from graphviz import Digraph

from src.data.dataset.nas.generation.darts import DartsCellArch, DartsNetworkArch
from src.data.preprocessing.utils import adj_matrix2edge_list


def test_cell_node_edge():

    cell_arch = DartsCellArch(num_inter_nodes=4)

    # edge_idx = 0
    edge_idx = 1783
    # edge_idx = cell_arch.num_all_edge_combs - 1

    # node_idx = 0
    node_idx = 27834
    # node_idx = cell_arch.num_all_op_combs - 1

    adj_matrix = cell_arch.sample_edges(edge_idx)
    node_list = cell_arch.sample_nodes(node_idx)

    edges_list = adj_matrix2edge_list(
        adj_matrix=adj_matrix,
        node_list=node_list
    )

    graph_dot = Digraph('round-table')
    graph_dot.edges(edges_list)
    graph_dot.render(filename='graph-darts-edge_%d-node_%d' % (edge_idx, node_idx))


def test_cell_arch():

    cell_arch = DartsCellArch(num_inter_nodes=4)

    num_arch = 10
    total_num = cell_arch.num_all_op_combs * cell_arch.num_all_edge_combs
    idx_samples = np.array([], dtype=int)

    while idx_samples.shape[0] < num_arch:
        new_samples = np.random.randint(low=0, high=total_num, size=num_arch - idx_samples.shape[0], dtype=int)
        idx_samples = np.unique(np.concatenate([idx_samples, new_samples]))

    for arch_idx in idx_samples:

        arch = cell_arch.sample_arch(arch_idx)

        edges_list = adj_matrix2edge_list(
            adj_matrix=arch['adj_matrix'],
            node_list=arch['node_list']
        )

        unique_nodes = set(arch['node_list'])

        graph_dot = Digraph('round-table', node_attr=dict(style='filled'))
        for node in unique_nodes:
            if 'INPUT' in node:
                graph_dot.node(node, shape='box', fillcolor='darkseagreen1')
            elif 'CONCAT' in node:
                graph_dot.node(node, shape='box', fillcolor='palegoldenrod')
            elif 'ADD' in node:
                graph_dot.node(node, shape='diamond', fillcolor='lightblue1')
            else:
                graph_dot.node(node, fillcolor='gray85')
        graph_dot.edges(edges_list)
        graph_dot.render(filename='graph-darts-arch_%d-edge_%d-node_%d'
                                  % (arch_idx, arch['edge_idx'], arch['node_idx']))


def test_network():

    cell_arch = DartsCellArch(num_inter_nodes=4)

    num_arch = 1
    total_num = cell_arch.num_all_op_combs * cell_arch.num_all_edge_combs

    architectures = {'normal': [], 'reduction': []}
    for k in architectures:
        idx_samples = np.array([], dtype=int)
        while idx_samples.shape[0] < num_arch:
            new_samples = np.random.randint(low=0, high=total_num, size=num_arch - idx_samples.shape[0], dtype=int)
            idx_samples = np.unique(np.concatenate([idx_samples, new_samples]))
        for arch_idx in idx_samples:
            arch = cell_arch.sample_arch(arch_idx)
            edges_list = adj_matrix2edge_list(
                adj_matrix=arch['adj_matrix'],
                node_list=arch['node_list']
            )

            architectures[k].append({
                'edges_list': edges_list,
                'arch_idx': arch_idx,
            })

    for normal_arch, reduction_arch in zip(architectures['normal'], architectures['reduction']):
        network = DartsNetworkArch(
            normal_arch=normal_arch['edges_list'],
            reduction_arch=reduction_arch['edges_list']
        )

        unique_nodes = []
        # print(network.edges_list)
        for node_i, node_j in network.edges_list:
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
        graph_dot.edges(network.edges_list)
        graph_dot.render(filename='graph-darts-network-norm_%d-red_%d'
                                  % (normal_arch['arch_idx'], reduction_arch['arch_idx']))


if __name__ == '__main__':
    test_network()
