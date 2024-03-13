import re
import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pickle
from src.data.preprocessing.utils import (
    remove_useless_node,
    GraphAdj,
    remove_loose_input
)
from src.config import Config
import json
# from src.data.dataset.nas.darts import get_unique_nodes, encode_edges, name2idx

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
        with open(path, mode) as fp:
            f_content = fp.read()
            data = json.loads(f_content)
    return data

def get_unique_ops(edges):
    # get unique_ops
    unique_ops = []
    for op_i, op_j in edges:
        op_i_ = re.findall('(.*)Backward', op_i)
        # encoder.layer
        if not op_i_ == []:
            unique_ops.append(op_i_[0])
        # else:
        #     unique_ops.append(op_i)
        op_j_ = re.findall('(.*)Backward', op_j)
        if not op_j_ == []:
            unique_ops.append(op_j_[0])
        # else:
        #     unique_ops.append(op_j)
    unique_ops = set(unique_ops)
    return unique_ops

def main():

    config = Config(default_config_file='configs/extract_ops.yaml')
    args = config.load_config()
    unique_ops = set()
    # unique_names = set()
    for task in args.tasks:
        paths_ = get_all_paths(os.path.join(args.root, task))
        paths = [file for file in paths_ if re.search('.*\.pkl$', file)]
        # print(paths)
        for path in paths:
            data = load_from_file(path, mode='rb')
            unique_ops_ = get_unique_ops(data['graph_edges'])
            unique_ops.update(unique_ops_)
            # unique_names.update(unique_names_)
    unique_ops = list(unique_ops)
    # unique_names = list(unique_names)
    ops2idx = {ops:idx+1 for idx, ops in enumerate(unique_ops)}
    ops2idx["unknown"] = 0
    # idx2ops = {idx: ops for idx, ops in enumerate(unique_ops)}

    json_map = json.dumps(ops2idx)
    file_name = os.path.join(args.output_root, 'ops2idx.json')
    with open(file_name, 'w') as f:
        f.write(json_map)
    
if __name__ == '__main__':
    main()
    