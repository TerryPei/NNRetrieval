import re
import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pickle
from src.config import Config
import json
from src.data.dataset.nas.darts import get_unique_nodes, encode_edges, name2idx

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

def get_unique_truth(name):
    # print(type(name))
    name_ = re.findall('.*?(resnet|vit|convnext|beit|\
                        mit|dog|pond|swin|llama|amee|flexivit|exper|\
                       autotrain|bert|mobilevit|mobile|roberta|bert|swin|roberta|\
                       RoBERTa|attenton|beit|yolo|BART|bart|gpt|\
                       xlnet|former|regnet|class|opus|swin|\
                       roberta|RoBERTa|beit|opus|deit|\
                       bart|gpt|t5|xls|wav2vec|xlnet|former|base|case|\
                       finetune).*?', name, re.IGNORECASE)

    if not name_ == []:
        return name_[0].lower()
    else:
        return 'unknown'

def get_unique_names(name):
    if re.findall('.*?(resnet|convnext|rust|regnet|class|animal|platzi|deit|dog|pond|conv|cnn).*?', name, re.IGNORECASE) != []:
        name_ = 'cnn-block'
    elif re.findall('.*?(vit|bert|Bert|swin|roberta|RoBERTa|att|beit\
                    |opus|deit|yolo|BART|bart|gpt|t5|xlm|xls|wav|xlnet|\
                    former|albert|-en-|_en_|base|case|finetune|disti|wav2vec).*?', name, re.IGNORECASE) != []:
        name_ = 'attention-block'
    else:
        name_ = 'unknown'
    return name_


def main():

    config = Config(default_config_file='configs/labels2idx.yaml')
    args = config.load_config()
    # unique_names = set()
    unique_names = set()
    task_names = set()

    unique_truth = set()

    for task in args.tasks:
        paths_ = get_all_paths(os.path.join(args.root, task))
        paths = [file for file in paths_ if re.search('.*\.pkl$', file)]
        # print(paths)
        
        min_params, max_params = 10000000, 0
        min_size, max_size = 10000000, 0
        min_flops, max_flops = 10000000, 0
        
        for path in paths:
            data = load_from_file(path, mode='rb')
            # name -> unique
            # unique_names_ = get_unique_names(data['model_name'])
            unique_names_ = get_unique_names(data['model_name'])
            unique_names.update({unique_names_})

            unique_truth_ = get_unique_truth(data['model_name'])
            unique_truth.update({unique_truth_})

            task_names.update({data["task_name"]})

            # params
            if max_params < data["model_params"]:
                max_params = data["model_params"]
            if min_params > data["model_params"]:
                min_params = data["model_params"]
            # size
            if max_size < data["model_size"]:
                max_size = data["model_size"]
            if min_size > data["model_size"]:
                min_size = data["model_size"]

            # flops
            if max_flops < data["flops"][0]:
                max_flops = data["flops"][0]
            if min_flops > data["flops"][0]:
                min_flops = data["flops"][0]


    labels2idx = {}
    # print(unique_names)
    unique_names = list(unique_names)
    unique_names.remove("unknown")
    labels2idx['model_names'] = {name: idx+1 for idx, name in enumerate(unique_names)}
    labels2idx['model_names']['unknown'] = 0

    labels2idx['model_params'] = {'max_params': max_params, 'min_params': min_params}
    labels2idx['model_size'] = {'max_size': max_size, 'min_size': min_size}
    labels2idx['model_flops'] = {'max_flops': max_flops, 'min_flops': min_flops}
    
    task_names = list(task_names)
    # print(task_names)
    labels2idx['task_names'] = {task: idx+1 for idx, task in enumerate(task_names)}
    # idx2labels = {idx: labels for idx, labels in enumerate(unique_names)}
    json_map = json.dumps(labels2idx)
    file_name = os.path.join(args.output_root, 'labels2idx.json')
    with open(file_name, 'w') as f:
        f.write(json_map)
    #### label2idx for pretraining ####

    #### name2truth_class_name for evaluation ####
    name2trueclass = {}
    unique_truth = list(unique_truth)
    unique_truth.remove("unknown")
    name2trueclass['model_names'] = {name: idx+1 for idx, name in enumerate(unique_truth)}
    name2trueclass['model_names']["unknown"] = 0

    json_map = json.dumps(name2trueclass)
    file_name = os.path.join(args.output_root, 'name2trueclass.json')
    with open(file_name, 'w') as f:
        f.write(json_map)

if __name__ == '__main__':
    main()
    