import functools
import itertools
import operator

import torch
import json
from src.data.preprocessing.utils import adj_matrix2edge_list
from .utils import base_convert

# DARTS_DEFAULT_OPERATIONS = (
#     # 'none',  # always outputs zeros, discarded
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5',
# )
with open('maps/ops2idx.json', 'r') as f:
    ops2idx = json.load(f)
# operations = (ops2idx.keys())
DARTS_DEFAULT_OPERATIONS = (
    ops2idx.keys()
)

DARTS_DEFAULT_STEM_CIFAR = (
    # ('stem.INPUT-0', 'stem.conv_3x3'),
    # ('stem.conv_3x3', 'stem.bn'),
    ops2idx.keys()
)

DARTS_DEFAULT_HEAD_CIFAR = (
    # ('head.adaptive_avg_pool_1x1', 'head.linear'),
    ops2idx.keys()
)


class DartsCellArch(object):

    def __init__(self, num_inter_nodes=4, op_candidates=DARTS_DEFAULT_OPERATIONS):
        super(DartsCellArch, self).__init__()

        self.num_inter_nodes = num_inter_nodes
        self.op_candidates = op_candidates

        self.num_inputs = 2
        self.num_outputs = 1
        self.num_edges_per_node = 2

        self.nodes_inputs = ['INPUT-%d' % i for i in range(self.num_inputs)]
        self.nodes_outputs = ['CONCAT-%d' % i for i in range(self.num_outputs)]

        self.num_total_nodes = (self.num_inputs + self.num_outputs) + self.num_inter_nodes * (self.num_edges_per_node + 1)
        self.num_operations = len(self.op_candidates)

        self.num_all_op_combs = self.num_operations ** (self.num_inter_nodes * self.num_edges_per_node)
        self.num_edges = [
            ((n + self.num_inputs) * (n + self.num_inputs - 1)) for n in range(self.num_inter_nodes)
        ]
        self.num_all_edge_combs = functools.reduce(operator.mul, self.num_edges, 1)

        self.operations_name2idx = {op: idx for idx, op in enumerate(self.op_candidates)}

        self._all_node_candidates = []
        self._edge_combs = None
        self._init_edge_combs()

    def _init_edge_combs(self):
        _count = 1

        for i in range(self.num_inter_nodes):
            node_candidates = [i for i in range(self.num_inputs)] + \
                              [self.num_inputs + (self.num_edges_per_node + 1) * (i + 1) - 1 for i in range(i)]
            self._all_node_candidates.append(node_candidates)

        self._edge_combs = [
            list(
                itertools.permutations(candidate, r=self.num_edges_per_node)
            )
            for candidate in self._all_node_candidates
        ]

    def sample_nodes(self, idx=None):
        if idx is None:
            raise NotImplementedError

        assert idx < self.num_all_op_combs, idx

        # sample nodes
        ops = base_convert(idx=idx, base=self.num_operations)
        if len(ops) < self.num_inter_nodes * self.num_edges_per_node:
            ops = [0] * (self.num_inter_nodes * self.num_edges_per_node - len(ops)) + ops
        try:
            nodes = [self.op_candidates[i] for i in ops]
        except IndexError as e:
            print(self.num_operations)
            print(ops)
            raise e

        for i in range(self.num_inter_nodes):
            nodes.insert(self.num_edges_per_node * (i + 1) + i, 'ADD')

        ops_count = {}
        new_nodes = []
        for op in nodes:
            if op in ops_count:
                ops_count[op] += 1
            else:
                ops_count[op] = 0
            new_nodes.append('%s-%d' % (op, ops_count[op]))

        nodes = self.nodes_inputs + new_nodes + self.nodes_outputs
        return nodes

    def sample_edges(self, idx=None):
        if idx is None:
            raise NotImplementedError

        assert idx < self.num_all_edge_combs, idx

        # calculate indices for each op
        indices = []
        for i in self.num_edges:
            indices.append(idx % i)
            idx //= i
        # print(indices)

        # sample adjacency matrix
        adj_matrix = torch.zeros((self.num_total_nodes, self.num_total_nodes))

        _count = 1
        _count_op = 0

        for i in range(self.num_inputs, self.num_total_nodes - self.num_outputs):
            if _count % (self.num_edges_per_node + 1) != 0:
                # op nodes
                # print(_count_op // self.num_edges_per_node, _count_op // self.num_edges_per_node, _count_op % self.num_edges_per_node)
                j = self._edge_combs[
                    _count_op // self.num_edges_per_node
                ][
                    indices[_count_op // self.num_edges_per_node]
                ][
                    _count_op % self.num_edges_per_node
                ]

                adj_matrix[j, i] = 1
                _count += 1
                _count_op += 1
            else:
                # the add node
                adj_matrix[i - 1, i] = 1
                adj_matrix[i - 2, i] = 1
                adj_matrix[i, -self.num_outputs:] = 1
                _count = 1

        return adj_matrix

    def sample_arch(self, idx=None):
        if idx is None:
            raise NotImplementedError

        node_idx = idx // self.num_all_edge_combs
        edge_idx = idx % self.num_all_edge_combs

        assert node_idx < self.num_all_op_combs, node_idx
        assert edge_idx < self.num_all_edge_combs, edge_idx

        return {
            'node_idx': node_idx,
            'node_list': self.sample_nodes(node_idx),
            'edge_idx': edge_idx,
            'adj_matrix': self.sample_edges(edge_idx)
        }

    def __iter__(self):
        self.arch_count = 0
        self.graph_count = []
        return self

    def __next__(self):
        if self.arch_count < self.num_all_op_combs * self.num_all_edge_combs:
            arch = self.sample_arch(idx=self.arch_count)
            self.arch_count += 1
            return arch
        else:
            raise StopIteration()


class DartsNetworkArch(object):

    def __init__(self, normal_arch, reduction_arch,
                 num_layers=20,
                 stem=DARTS_DEFAULT_STEM_CIFAR,
                 head=DARTS_DEFAULT_HEAD_CIFAR):

        super(DartsNetworkArch, self).__init__()

        if isinstance(normal_arch, dict):
            normal_edges_list = adj_matrix2edge_list(
                adj_matrix=normal_arch['adj_matrix'],
                node_list=normal_arch['node_list'],
            )
        elif isinstance(normal_arch, list):
            normal_edges_list = normal_arch
        else:
            raise ValueError(type(normal_arch))

        if isinstance(reduction_arch, dict):
            reduction_edges_list = adj_matrix2edge_list(
                adj_matrix=reduction_arch['adj_matrix'],
                node_list=reduction_arch['node_list'],
            )
        elif isinstance(reduction_arch, list):
            reduction_edges_list = reduction_arch
        else:
            raise ValueError(type(reduction_arch))

        network_edges_list = []
        i = -1
        for i in range(num_layers):
            if i in [num_layers // 3, 2 * num_layers // 3]:
                # reduction
                for node_i, node_j in reduction_edges_list:
                    if 'INPUT-0' in node_i:
                        if i >= 2:
                            node_i = 'layer-%d.CONCAT-0' % (i - 2,)
                        else:
                            node_i = 'stem.bn'
                    elif 'INPUT-1' in node_i:
                        if i >= 1:
                            node_i = 'layer-%d.CONCAT-0' % (i - 1,)
                        else:
                            node_i = 'stem.bn'
                    else:
                        node_i = 'layer-%d.%s' % (i, node_i)
                    node_j = 'layer-%d.%s' % (i, node_j)
                    network_edges_list.append((node_i, node_j))
            else:
                # normal
                for node_i, node_j in normal_edges_list:
                    if 'INPUT-0' in node_i:
                        if i >= 2:
                            node_i = 'layer-%d.CONCAT-0' % (i - 2,)
                        else:
                            node_i = 'stem.bn'
                    elif 'INPUT-1' in node_i:
                        if i >= 1:
                            node_i = 'layer-%d.CONCAT-0' % (i - 1,)
                        else:
                            node_i = 'stem.bn'
                    else:
                        node_i = 'layer-%d.%s' % (i, node_i)
                    node_j = 'layer-%d.%s' % (i, node_j)
                    network_edges_list.append((node_i, node_j))

        network_edges_list.append((
            'layer-%d.CONCAT-0' % (i,),
            'head.adaptive_avg_pool_1x1'
        ))

        self.edges_list = list(stem) + network_edges_list + list(head)
