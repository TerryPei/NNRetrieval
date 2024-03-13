import logging
import os

import torch
import json
from src import utils
from src.config import Config
from src.data.dataset.nas.darts import DartsSubgraphDatasetInMem, DartsGraphDataset, DartsSubgraphDatasetFolder
from src.training.trainer import SubgraphTrainer, GraphTrainer
from src.utils import set_output_dir, set_logging


def main():
    config = Config(default_config_file='configs/darts_pretraining.yaml')
    args = config.load_config()

    args.save_dir = set_output_dir(save='-'.join([args.save_task_name, args.save_exp_name]))
    set_logging(save=args.save_dir)
    args.dataset_graph_path = os.path.join(args.save_dir, args.dataset_graph_path)

    logging.info('Arguments: %s' % (args,))

    if os.path.isfile(args.dataset_subgraph_path) and 'csv' == args.dataset_subgraph_path[-3:]:
        darts_subgraph_dataset = DartsSubgraphDatasetInMem(root=args.dataset_subgraph_path)
    elif os.path.isdir(args.dataset_subgraph_path):
        darts_subgraph_dataset = DartsSubgraphDatasetFolder(root=args.dataset_subgraph_path) # !
    else:
        raise RuntimeError(str(args.dataset_subgraph_path))

    logging.info('DARTS subgraph dataset: %d samples' % len(darts_subgraph_dataset))
    
    with open('maps/ops2idx.json', 'r') as f:
        ops2idx = json.load(f)
    # print(max(ops2idx.values())+1)
    args.subgraph_model_in_channels = max(ops2idx.values())+1

    subgraph_trainer = SubgraphTrainer(
        model_args={
            'in_channels': args.subgraph_model_in_channels,
            'hidden_channels': args.subgraph_model_hidden_channels,
            'num_layers': args.subgraph_model_num_layers,
        },
        dataset=darts_subgraph_dataset, # !
        subgraph_lr=args.subgraph_model_subgraph_lr,
        context_lr=args.subgraph_model_context_lr,
        batch_size=args.subgraph_model_bs,
        save_dir=args.save_dir,
        logging_interval=args.logging_interval,
        device=args.device,
    )
    subgraph_params = utils.count_parameters(subgraph_trainer.subgraph_model)
    context_params = utils.count_parameters(subgraph_trainer.context_model)
    logging.info('subgraph_model params: %.f' % (subgraph_params,))
    logging.info('context_model params: %.f' % (context_params,))

    for ep in range(args.subgraph_model_epoch):
        subgraph_trainer.step()

    # exit(0)
    # extract macro data
    subgraph_trainer.build_macro_graph(file_name=args.dataset_graph_path)
    darts_graph_dataset = DartsGraphDataset(root=args.dataset_graph_path)

    logging.info('DARTS (macro) graph dataset: %d samples' % len(darts_graph_dataset))
    with open('maps/labels2idx.json', 'r') as f:
        labels2idx = json.load(f)
    args.graph_model_num_classes = len(labels2idx['model_names'].values())+1
    macro_arch_trainer = GraphTrainer(
        model_args={
            'in_channels': args.graph_model_in_channels,
            'hidden_channels': args.graph_model_hidden_channels,
            'num_layers': args.graph_model_num_layers,
            'num_regs': args.graph_model_num_regs,
            'num_classes': args.graph_model_num_classes,
        },
        dataset=darts_graph_dataset,
        lr=args.graph_model_lr,
        batch_size=args.graph_model_bs,
        logging_interval=args.logging_interval,
        device=args.device,
    )

    macro_model_params = utils.count_parameters(macro_arch_trainer.model)
    logging.info('(macro) model params: %.f' % (macro_model_params,))
    
    best_acc = 0
    for ep in range(args.graph_model_epoch):
        acc = macro_arch_trainer.step()
        if acc > best_acc:
            best_acc = acc
            # if not os.path.isdir(args.save_dir):
            #     os.mkdir(args.cpkt_model_path)
            model_to_save = macro_arch_trainer.model.module if hasattr(macro_arch_trainer.model,'module') else macro_arch_trainer.model
            cpkt = {
                # 'epoch': ep,
                'encoder': model_to_save.encoder.state_dict(),
                'classifer': model_to_save.classifer.state_dict(),
                'reg': model_to_save.reg.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict()
            }
            model_path = os.path.join(args.save_dir, 'pretrained_model.pt')
            torch.save(cpkt, model_path)
            # logging.info("Update the checkpoints on epoch {} with acc {}.".format(ep, acc))

    embeddings = macro_arch_trainer.get_graph_embedding()
    logging.info('%s' % (embeddings.size(),))

    torch.save(embeddings, os.path.join(args.save_dir, 'embeddings.pt'))


if __name__ == '__main__':
    main()


