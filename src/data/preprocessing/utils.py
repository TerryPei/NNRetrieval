import numpy as np
import torch
import json, re

def adj_matrix2edge_list(adj_matrix, node_list):
    edges_list = []

    for i, j in zip(*torch.where(adj_matrix != 0)):
        edges_list.append((node_list[i], node_list[j]))

    return edges_list


def remove_useless_node(graph_edges, node_name='AccumulateGrad'):
    new_graph_edges = []

    for node_i, node_j in graph_edges:

        if node_name not in node_i and node_name not in node_j:
            new_graph_edges.append([node_i, node_j])

        if node_name in node_j:
            for node_p, node_q in graph_edges:
                if node_j == node_p:
                    new_graph_edges.append([node_i, node_q])

    return new_graph_edges

def get_loose_input(graph_edges):
    # loose_input = [graph.nodes_list[_i] for _i in np.where(graph.in_degree == 0)[0]] # graph structure change
    visited, loose_input = [], []
    for node_i, node_j in graph_edges:
        visited.append(node_j)
    for node_i, node_j in graph_edges:
        if not node_i in visited:
            loose_input.append(node_i)
    return loose_input


def remove_loose_input(graph_edges, loose_input):
    # find nodes connected to a loose input, they need updates
    nodes_to_update = {}
    for node_i, node_j in graph_edges:
        if node_i in loose_input:
            if node_j not in nodes_to_update:
                nodes_to_update[node_j] = '+'.join([node_j, node_i])
            else:
                nodes_to_update[node_j] = '&'.join([nodes_to_update[node_j], node_i])
        if node_j in loose_input:
            raise ValueError('%s is not a loose input' % node_j)

    new_graph_edges = []
    for node_i, node_j in graph_edges:
        # if not a loose input, add to new graph
        if node_i not in loose_input:
            # check if need update
            if node_i in nodes_to_update:
                node_i = nodes_to_update[node_i]
            if node_j in nodes_to_update:
                node_j = nodes_to_update[node_j]
            # add node to new graph
            new_graph_edges.append([node_i, node_j])

    return new_graph_edges


def convert_nodes2idx(node, ops2idx):
    op = re.findall('(.*?)Backward.?', node)
    # node2ops = [x[0] for x in node2ops_ if x != []]
    if op == []:
        return ops2idx["unknown"] # unknown ops, but should be unique.
    else:
        return ops2idx[op[0]]

class GraphAdj(object):

    def __init__(self, graph_edges, type='generation'):

        super().__init__()
        # get nodes
        all_nodes = []

        for node_i, node_j in graph_edges:
            all_nodes.append(node_i)
            all_nodes.append(node_j)
        ################################################################
        #                                                              #
        #                     This the NasNet                          #
        #                                                              #
        ################################################################
        # print('%d nodes with redundancy' % len(all_nodes))
        all_nodes = set(all_nodes)
        num_nodes = len(all_nodes) # !
        # print('%d unique nodes in graph' % num_nodes)

        # build adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes))

        ######nas adj graph
        mapping_node_name2idx = {}
        _curr_max_idx = 0

        for node_i, node_j in graph_edges:
            if node_i not in mapping_node_name2idx:
                assert _curr_max_idx < num_nodes
                mapping_node_name2idx[node_i] = _curr_max_idx
                _curr_max_idx += 1
            node_idx_i = mapping_node_name2idx[node_i]

            if node_j not in mapping_node_name2idx:
                assert _curr_max_idx < num_nodes
                mapping_node_name2idx[node_j] = _curr_max_idx
                _curr_max_idx += 1
            node_idx_j = mapping_node_name2idx[node_j]
            adj_matrix[node_idx_i, node_idx_j] = 1

        mapping_node_idx2name = {}
        nodes_list = []
        for k in mapping_node_name2idx:
            mapping_node_idx2name[mapping_node_name2idx[k]] = k
            nodes_list.append(k)

        #### real net adj graph
        ################################################################
        #                                                              #
        #                     This the RealNet                         #
        #                                                              #
        ################################################################
        # all_nodes = set(all_nodes)
        # num_nodes = len(all_nodes)
        # # if type == 'generation':
        # #     with open('maps/ops2idx.json', 'r') as f:
        # #         operations2idx = json.load(f)
        # # else:
        # #     with open('maps/ops2idx_for_pretrain.json', 'r') as f:
        # #         operations2idx = json.load(f)  
        # # with open('maps/ops2idx.json', 'r') as f:
        # #     operations2idx = json.load(f)     
        # # mapping_node_name2idx = dict()

        # # count = max(operations2idx.values())+1 # nodes not in BackWard but also should be unique.

        # # for i, node in enumerate(all_nodes):
        # #     idx = convert_nodes2idx(node, operations2idx)
        # #     if idx == operations2idx["unknown"]:
        # #         mapping_node_name2idx[node] = count
        # #         count += 1
        # #     else:
        # #         mapping_node_name2idx[node] = idx
        # mapping_node_name2idx = {node: idx for idx, node in enumerate(all_nodes)}
        # # mapping_node_name2idx = {node: convert_nodes2idx(node, operations2idx, count+i) for i, node in enumerate(all_nodes)}
        # # mapping_node_idx2name = {convert_nodes2idx(node, operations2idx, count+i): node for i, node in enumerate(all_nodes)}
        # # num_nodes = len(mapping_node_name2idx)
        # max_node_id = max(mapping_node_name2idx.values())+1
        # # max_node_id = len(mapping_node_name2idx)

        # # print("num_nodes: ", num_nodes, "max_node_id: ", max_node_id)
        # # print("len(mapping_node_name2idx): ", len(mapping_node_name2idx))
        # # print("max(mapping_node_name2idx.values()): ", max(mapping_node_name2idx.values()))
        # # print('%d unique nodes in graph' % num_nodes)
        # # build adjacency matrix
        # adj_matrix = np.zeros((max_node_id, max_node_id))

        # for node_i, node_j in graph_edges:
        #     node_idx_i = mapping_node_name2idx[node_i]
        #     node_idx_j = mapping_node_name2idx[node_j]
        #     # print(node_idx_i, node_idx_j)
        #     adj_matrix[node_idx_i, node_idx_j] = 1
        
        # # nodes_list = []
        # # for k in mapping_node_name2idx:
        # #     nodes_list.append(k)
        # mapping_node_idx2name = {}
        # nodes_list = []
        # for k in mapping_node_name2idx:
        #     mapping_node_idx2name[mapping_node_name2idx[k]] = k
        #     nodes_list.append(k)
        # print(nodes_list)

        self.graph_edges = graph_edges
        # self.num_nodes = num_nodes
        self.adj_matrix = adj_matrix

        self.mapping_node_name2idx = mapping_node_name2idx
        # print(self.mapping_node_name2idx)
        self.mapping_node_idx2name = mapping_node_idx2name
        # print(self.mapping_node_idx2name)
        self.nodes_list = nodes_list

    # @property
    # def in_degree(self):
    #     in_degree = dict()
    #     visited = dict()
    #     for node_i, node_j in self.graph_edges:
    #         # graph_edges
    #         if node_j in visited:
    #             in_degree[node_j] += 1
    #         else:
    #             in_degree[node_i] = 0
    #             in_degree[node_j] = 1

    # @property
    # def out_degree(self):
    #     out_degree = dict()
    #     visited = dict()
    #     for node_i, node_j in self.graph_edges:
    #         if node_j in visited:
    #             out_degree[node_i] += 1
    #         else:
    #             out_degree[node_i] = 0
    #             out_degree[node_i] = 1
    @property
    def in_degree(self):
        return self.adj_matrix.sum(0)

    @property
    def out_degree(self):
        return self.adj_matrix.sum(-1)
