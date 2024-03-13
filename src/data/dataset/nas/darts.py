import csv
import json
import os

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from src.data.dataset.nas.generation.darts import DARTS_DEFAULT_OPERATIONS
from src.data.dataset.utils import build_subgraph_context_pairs, build_macro_graph
import re
import json


        
def get_unique_nodes(edges):
    # get unique_nodes
    unique_nodes = []
    for node_i, node_j in edges:
        unique_nodes.append(node_i)
        unique_nodes.append(node_j)
    unique_nodes = set(unique_nodes)
    return unique_nodes

def encode_nodes(unique_nodes, operations2idx):
    # encode nodes
    node_feat = []
    for node in unique_nodes:
        unknown = True
        for op in operations2idx:
            if op in node:
                node_feat.append(operations2idx[op])
                unknown = False
                break
        assert unknown is False, (node, operations2idx.keys())
    assert len(node_feat) == len(unique_nodes), (len(node_feat), len(unique_nodes))
    # print(len(node_feat)) # dynamic
    # print(len(operations2idx)) # 28 fixed
    # print(operations2idx)
    # print(type(node_feat[0]), len(node_feat))s
    # exit(0)
    x = F.one_hot(
        torch.tensor(node_feat),
        num_classes=len(operations2idx)
    ).to(torch.float32)
    # print(x.shape) #[len(node_feat), len(operations2idx)] = [num_node_feat, num_classes] each node is also an operation
    return x


def encode_edges(edges, node2idx):
    # encode edges
    edge_index = [[], []]
    for node_i, node_j in edges:
        edge_index[0].append(node2idx[node_i])
        edge_index[1].append(node2idx[node_j])
    edge_index = torch.tensor(edge_index)
    return edge_index

def convert_nodes2idx(node, ops2idx):
    op = re.findall('(.*?)Backward.?', node)
    # node2ops = [x[0] for x in node2ops_ if x != []]
    # print(op[0])
    if op == []:
        return ops2idx["unknown"] # unknown ops, but should be unique.
    else:
        return ops2idx[op[0]]
    
def name2idx(model_name, labels2idx):
    if model_name in labels2idx['model_names'].keys():
        return labels2idx['model_names'][model_name]
    else:
        return 0

def task2idx(task, labels2idx):
    return labels2idx['task_names'][task]

with open('maps/labels2idx.json', 'r') as f:
    data = json.load(f)
unique_truth = list(data["model_names"].keys())

def get_unique_names(name):
    # print(type(name))
    name_ = re.findall('.*?('+'|'.join(unique_truth)+').*?', name, re.IGNORECASE)

    if not name_ == []:
        return name_[0].lower()
    else:
        return 'unknown'

class DartsSubgraphDatasetInMem(Dataset):
    """In-memory (CSV) version dataset"""

    def __init__(self, root, operations=DARTS_DEFAULT_OPERATIONS):
        super(DartsSubgraphDatasetInMem, self).__init__()
        self.root = root

        # self.operations = ['stem', 'head', 'INPUT', 'CONCAT', 'ADD'] + list(operations)
        # self.operations2idx = {op: idx for idx, op in enumerate(self.operations)}

        with open('maps/ops2idx.json', 'r') as f:
            ops2idx = json.load(f)
        self.operations = list((ops2idx.keys())) #+ ['stem', 'head', 'INPUT', 'CONCAT', 'ADD'] + list(operations)
        # self.operations2idx = {op: idx for idx, op in enumerate(self.operations)}
        self.operations2idx = ops2idx

        self.graphs = self.load_from_file(root=self.root)  # raw graph
        self.batch = [None] * len(self.graphs)  # batch cache
        self.macro_arch_graph = [None] * len(self.graphs)  # macro_arch_graph cache

    @staticmethod
    def preprocessing(raw_data):
        graphs = [[]]  # save each raw graphs
        prev_graph_idx = None

        for row in raw_data:
            # read from row
            graph_idx, node_i, node_j = row[0], row[1], row[2]
            # print(graph_idx, node_i, node_j)
            # save to all edges
            node_i = '.'.join([graph_idx, node_i])
            node_j = '.'.join([graph_idx, node_j])

            if prev_graph_idx is None:
                prev_graph_idx = graph_idx

            if graph_idx != prev_graph_idx:  # is a new graph
                graphs.append([(node_i, node_j)])
                prev_graph_idx = graph_idx
            else:  # otherwise
                graphs[-1].append((node_i, node_j))

        return graphs

    @staticmethod
    def load_from_file(root):

        with open(root, 'r') as fp:
            reader = csv.reader(fp)
            reader_iter = iter(reader)
            next(reader_iter)  # omit header

            graphs = DartsSubgraphDatasetInMem.preprocessing(reader_iter)
        # print("check1")
        return graphs  # , data, slices, node2idx


    @staticmethod
    def build_batch(edges, operations2idx):
        # print(edges)
        raw_pairs = build_subgraph_context_pairs(graph_edges=edges, with_remove=True) # edges -> ['subgraph', 'pos_context', 'neg_context']
        
        macro_arch_graph = build_macro_graph(list_subgraph=raw_pairs['subgraph'], n_type=int)

        data_pairs = {}
        for k in raw_pairs:
            # save data in list
            list_batch = []
            for es in raw_pairs[k]:
                # get unique nodes
                unique_nodes = get_unique_nodes(edges=es)
                # mapping nodes to index
                node2idx = {node: idx for idx, node in enumerate(unique_nodes)}
                x = encode_nodes(unique_nodes=unique_nodes, operations2idx=operations2idx)
                edge_index = encode_edges(edges=es, node2idx=node2idx)
                # append to batch
                list_batch.append(Data(x=x, edge_index=edge_index))

            try:
                data_pairs[k] = Batch.from_data_list(list_batch)
            except RuntimeError as e:
                print(list_batch)
                raise e

        return data_pairs, macro_arch_graph

    def __getitem__(self, index):

        if self.batch[index] is None or self.macro_arch_graph[index] is None:
            edges = self.graphs[index]
            batch, macro_arch_graph = self.build_batch(
                edges=edges,
                operations2idx=self.operations2idx
            )

            self.batch[index] = batch
            self.macro_arch_graph[index] = macro_arch_graph

            return self.batch[index], self.macro_arch_graph[index]
        else:
            return self.batch[index], self.macro_arch_graph[index]

    def __len__(self):
        return len(self.graphs)


class DartsSubgraphDatasetFolder(Dataset):
    """Dynamic-loading (JSON) version dataset"""
    def __init__(self, root, operations=DARTS_DEFAULT_OPERATIONS):
        self.root = root
        # self.operations = ['stem', 'head', 'INPUT', 'CONCAT', 'ADD'] + list(operations)
        # self.operations2idx = {op: idx for idx, op in enumerate(self.operations)}
        with open('maps/ops2idx.json', 'r') as f:
            ops2idx = json.load(f)
        self.operations = list((ops2idx.keys())) #+ ['stem', 'head', 'INPUT', 'CONCAT', 'ADD'] + list(operations)
        # self.operations2idx = {op: idx for idx, op in enumerate(self.operations)}
        self.operations2idx = ops2idx
        # print(self.root) # drive prob of in/out read data here, just relaunch
        paths = os.listdir(self.root)
        self.paths = [os.path.join(self.root, p) for p in paths]
        with open('maps/labels2idx.json', 'r') as f:
            self.labels2idx =  json.load(f)

    @staticmethod
    def load_from_file(path):
        with open(path, 'r') as fp:
            f_content = fp.read()
            data = json.loads(f_content)
        return data

    def __getitem__(self, index):
        path = self.paths[index]
        data = self.load_from_file(path)
        unique_nodes = data['unique_nodes']
        edge_index = data['edge_index']

        edges = []
        for i in range(len(edge_index[0])):
            node_i = unique_nodes[edge_index[0][i]]
            node_j = unique_nodes[edge_index[1][i]]
            edges.append((node_i, node_j))
        
        batch, macro_arch_graph = DartsSubgraphDatasetInMem.build_batch(
            edges=edges,
            operations2idx=self.operations2idx,

        )
        
        r = {}
        
        model_name = get_unique_names(data['model_name'])
        r['model_name'] = name2idx(model_name, self.labels2idx)

        r['task_name'] = task2idx(data['task_name'], self.labels2idx)

        r['model_flops'] = data['flops'][0]/(self.labels2idx['model_flops']['max_flops'] - self.labels2idx['model_flops']['min_flops'])
        r['model_size'] = data['model_size']/(self.labels2idx['model_size']['max_size'] - self.labels2idx['model_size']['min_size'])
        r['model_params'] = data['model_params']/(self.labels2idx['model_params']['max_params'] - self.labels2idx['model_params']['min_params'])


        cls_label = torch.tensor(r['model_name'])
        reg_label = torch.tensor([r['model_flops'], r['model_size'], r['model_params']]).reshape(1, 3)
        
        label_class = torch.tensor(data['label_class'])
        # gruond_truth_class = 
        return batch, macro_arch_graph, cls_label, reg_label, label_class

    def __len__(self):
        return len(self.paths)


class DartsGraphDataset(object):

    def __init__(self, root):
        super(DartsGraphDataset, self).__init__()
        self.data = DartsGraphDataset._load_from_file(root)

    @staticmethod
    def _load_from_file(root):
        raw_data = torch.load(root)
        data = []
        for p in raw_data:
            # print(p.keys()) # right
            # exit(0)
            data.append(Data(
                x=p['x'],
                edge_index=p['edge_index'],
                cls_label = p['cls_label'],
                reg_label = p['reg_label'],
                label_class = p['label_class'],
            ))
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
