import csv
import json
import os
from typing import List
import numpy as np
from tqdm import tqdm
import pickle
from src.config import Config
from src.data.dataset.nas.darts import get_unique_nodes, encode_edges, name2idx
from src.data.preprocessing.utils import (
    remove_useless_node,
    get_loose_input,
    GraphAdj,
    remove_loose_input
)
from scanner.labels2idx import get_unique_names
from scanner.labels2idx import get_unique_truth
import torch
import re

def get_all_paths(root):
    paths = os.listdir(root)
    paths = [os.path.join(root, p) for p in paths]
    return paths

def load_from_file(path, mode='rb'):
    # print(path)
    mode = 'rb' if path.endswith('.pkl') else 'r'
    if mode == 'rb':
        with open(path, mode) as fp:
            data = pickle.load(fp)
    else:
        with open(path, 'r') as fp:
            f_content = fp.read()
            data = json.loads(f_content)
    return data 

def get_unique_nodes(edges):
    # get unique_nodes
    unique_nodes = []
    for node_i, node_j in edges:
        unique_nodes.append(node_i)
        unique_nodes.append(node_j)
    unique_nodes = set(unique_nodes)
    return unique_nodes

def encode_edges(edges, node2idx):
    # encode edges
    edge_index = [[], []]
    for node_i, node_j in edges:
        edge_index[0].append(node2idx[node_i])
        edge_index[1].append(node2idx[node_j])
    edge_index = torch.tensor(edge_index)
    return edge_index

def load_real_arch(root: str):
    task_paths_ = get_all_paths(root)
    task_paths = [file for file in task_paths_ if not re.search('.*\.DS_Store$', file)]
    # print(task_paths)
    # exit(0)
    for task in task_paths:
        paths_ = get_all_paths(task)
        paths = [file for file in paths_ if re.search('.*\.pkl$', file)]
        for arch_idx, p in enumerate(paths):
            data = load_from_file(p)  # TODO
            graph_edges = data['graph_edges']
            # remove things
            graph_edges = remove_useless_node(graph_edges=graph_edges, node_name='AccumulateGrad')
            graph_edges = remove_useless_node(graph_edges, node_name='TBackward')
            # graph = GraphAdj(graph_edges=graph_edges)
            # loose_input = [graph.nodes_list[_i] for _i in np.where(graph.in_degree == 0)[0]] # graph structure change
            loose_input = get_loose_input(graph_edges)
            graph_edges = remove_loose_input(graph_edges, loose_input)
            data['graph_edges'] = graph_edges
            # Change the adj graph
            # yield arch_name, edges_list
            yield data

def convert_nodes2idx(node, ops2idx):
    op = re.findall('(.*)Backward', node)
    # node2ops = [x[0] for x in node2ops_ if x != []]
    if op == []:
        return ops2idx["unknown"] # unknown ops, but should be unique.
    else:
        return ops2idx[op[0]]




def main():
    config = Config()
    config.parser.add_argument('--output', type=str, default='data')
    config.parser.add_argument('--num_arch', type=int, default=100)
    config.parser.add_argument('--format', type=str, default='json', choices=['csv', 'json'])
    config.parser.add_argument('--root', type=str, default='datasets')
    args = config.load_config()

    network_graphs_iter = load_real_arch(args.root)

    if 'csv' == args.format:
        header = ['graph_index', 'node_i', 'node_j']
        path = os.path.join(
            args.output,
            'darts-%d.csv' % args.num_arch
        )
        with open(path, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            for i, g in enumerate(network_graphs_iter):
                for node_i, node_j in g[-1]:
                    writer.writerow([i, node_i, node_j])

    elif 'json' == args.format:
        path = os.path.join(
            args.output,
            'darts-json-%d' % args.num_arch,
        )
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            raise RuntimeError('%s exists' % path)
        
        with open('maps/labels2idx.json', 'r') as f:
            labels2idx = json.load(f)

        with open('maps/name2trueclass.json', 'r') as f:
            name2trueclass = json.load(f)

        with tqdm(total=args.num_arch) as p_bar:
            p_bar.set_description('Saving to JSON:')
            # TODO:
            
            for data in network_graphs_iter:
                # get unique nodes
                edges_list = data['graph_edges']
                unique_nodes = get_unique_nodes(edges=edges_list)
                # mapping nodes to index based on the map dictionary
                # node2ops = [re.findall('(.*)Backward', node) for node in unique_nodes]
                # with open('maps/ops2idx.json', 'r') as f:
                #     ops2idx = json.load(f)
                # # count = len(unique_nodes) + 1
                # node2idx = {node: convert_nodes2idx(node, ops2idx, count+i) for i, node in enumerate(unique_nodes)}
                node2idx = {node: idx for idx, node in enumerate(unique_nodes)}
                # node2idx = dict()
                # count = max(ops2idx.values())+1 # nodes not in BackWard but also should be unique.
                # for i, node in enumerate(unique_nodes):
                #     idx = convert_nodes2idx(node, ops2idx)
                #     if idx == ops2idx["unknown"]:
                #         node2idx[node] = count
                #         count += 1
                #     else:
                #         node2idx[node] = idx
                # node2idx = {node: ops2idx[node2ops[node]] for node in unique_nodes}
                edge_index = encode_edges(edges=edges_list, node2idx=node2idx)
                # dump as json
                g_json = json.dumps({
                    'model_name': data['model_name'],
                    'model_size': data['model_size'],
                    'model_params': data['model_params'],
                    'task_name': data['task_name'],
                    'inputs_shape': data['inputs_shape'],
                    'flops': data['flops'],
                    'unique_nodes': list(unique_nodes),
                    'edge_index': edge_index.tolist(),
                    'label_class': name2idx(get_unique_names(data['model_name']), labels2idx),
                    'true_class': name2idx(get_unique_truth(data['model_name']), name2trueclass), 
                })
                # save to files
                f_name = os.path.join(path, 'graph-%s.json' % (data['model_name']))
                with open(f_name, 'w') as fp:
                    fp.write(g_json)
                # break
                p_bar.update(1)

if __name__ == '__main__':
    main()
