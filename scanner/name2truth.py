import re
import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pickle
from src.config import Config
import json
from src.data.dataset.nas.darts import get_unique_nodes, encode_edges, name2idx
from scanner.labels2idx import get_unique_truth, get_unique_names

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
            data = json.load(fp)
    return data

def main():

    config = Config(default_config_file='configs/name2truth.yaml') 
    args = config.load_config()

    # file_name = os.path.join(args.output_root, 'labels2idx.json')
    file_name = os.path.join(args.output_root, 'name2trueclass.json')
    with open(file_name, 'r') as f:
        name2trueclass = json.load(f)
    # print(labels2idx['model_name'])
    class2truth = {}

    paths_ = get_all_paths(args.root)
    models = [file for file in paths_ if re.search('.*\.json$', file)]

    for idx, model in enumerate(models):
        data = load_from_file(model, mode='r')
        # print(data['model_name'])
        model_class = name2idx(get_unique_names(data['model_name']), name2trueclass) 
        
        if model_class in class2truth:
            class2truth[model_class].append(idx)
        else:
            class2truth[model_class] = [idx,]

    json_map = json.dumps(class2truth)
    
    file_name = os.path.join(args.output_root, 'class2truth.json')
    with open(file_name, 'w') as f:
        f.write(json_map)
    
    
if __name__ == '__main__':
    main()