import logging
import os
import time
import torch.nn.functional as F
import torch
from torch import optim, nn
from torch_geometric.data import DataLoader # !
from tqdm import tqdm
from src.nn.model import GraphNetwork, SubgraphDiscriminator, MLP
from src.training.utils import binary_accuracy
from sklearn.metrics import accuracy_score
from src.utils import AverageMeter


class SubgraphTrainer(object):
    
    def __init__(self, model_args, dataset, subgraph_lr, context_lr,
                 batch_size=2, optimizer_kwargs=None, save_dir=None, labeled=False,
                 logging_interval=50, device='cuda'):
        super(SubgraphTrainer, self).__init__()

        self.labeled = labeled

        # build model
        self.subgraph_model = GraphNetwork(  # used to generate the final embedding
            in_channels=model_args['in_channels'],
            hidden_channels=model_args['hidden_channels'],
            num_layers=model_args['num_layers'],
            **model_args.get('kwargs', {}),
        ).to(device)
        self.context_model = GraphNetwork(  # for pre-training only
            in_channels=model_args['in_channels'],
            hidden_channels=model_args['hidden_channels'],
            num_layers=model_args['num_layers'],
            **model_args.get('kwargs', {}),
        ).to(device)
        self.subgraph_discriminator = SubgraphDiscriminator(  # predict whether same source
            embed_dim=model_args['hidden_channels'] * 2
        ).to(device)

        # binary classification loss
        self.criterion = nn.BCELoss()

        # dataset
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        # set optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.subgraph_optimizer = optim.Adam(self.subgraph_model.parameters(),
                                             lr=subgraph_lr, **optimizer_kwargs)
        self.context_optimizer = optim.Adam(self.context_model.parameters(),
                                            lr=context_lr, **optimizer_kwargs)

        self.epoch = 0

        self.save_dir = save_dir
        self.logging_interval = logging_interval
        self.device = device

    def step(self, logging_interval=None):

        self.epoch += 1

        logging_interval = self.logging_interval if logging_interval is None else logging_interval

        start = time.time()

        # set to training mode
        self.subgraph_model.train()
        self.context_model.train()
        self.subgraph_model.set_return_embedding()
        self.context_model.set_return_embedding()
        self.subgraph_discriminator.train()
        # self.data_loader.dataset.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        time_meter = AverageMeter()

        num_batches = len(self.data_loader)
        for i, (pairs, _, _, _, _) in enumerate(self.data_loader):
            batch_start = time.time()

            self.subgraph_optimizer.zero_grad()
            self.context_optimizer.zero_grad()

            pairs['subgraph'] = pairs['subgraph'].to(self.device)
            pairs['pos_context'] = pairs['pos_context'].to(self.device)
            pairs['neg_context'] = pairs['neg_context'].to(self.device)

            # print(pairs['subgraph'])
            # print(pairs['pos_context'])
            # print(pairs['neg_context'])

            subgraph_vec = self.subgraph_model(
                x=pairs['subgraph'].x,
                edge_index=pairs['subgraph'].edge_index,
                batch=pairs['subgraph'].batch,
            )
            pos_context_vec = self.context_model(
                x=pairs['pos_context'].x,
                edge_index=pairs['pos_context'].edge_index,
                batch=pairs['pos_context'].batch,
            )
            neg_context_vec = self.context_model(
                x=pairs['neg_context'].x,
                edge_index=pairs['neg_context'].edge_index,
                batch=pairs['neg_context'].batch,
            )



            out_pos = self.subgraph_discriminator(subgraph_vec=subgraph_vec, context_vec=pos_context_vec)
            out_neg = self.subgraph_discriminator(subgraph_vec=subgraph_vec, context_vec=neg_context_vec)


            loss = self.criterion(out_pos, torch.ones_like(out_pos)) \
                   + self.criterion(out_neg, torch.zeros_like(out_neg))

            loss.backward()

            self.subgraph_optimizer.step()
            self.context_optimizer.step()

            loss_meter.update(loss.item(), n=out_pos.size(0)+out_neg.size(0))
            acc_meter.update(binary_accuracy(pred=out_pos, target=torch.ones_like(out_pos)), n=out_pos.size(0))
            acc_meter.update(binary_accuracy(pred=out_neg, target=torch.zeros_like(out_neg)), n=out_neg.size(0))
            time_meter.update(time.time() - batch_start, n=1)

            # logging
            if i % logging_interval == 0:
                logging.info(
                    '  epoch %d batch %d/%d (%.3fs/batch): loss=%.2f (%.2f), avg_acc=%.2f'
                    % (self.epoch, i, num_batches, time_meter.avg,
                       loss.item(), loss_meter.avg, acc_meter.avg)
                )

        # save
        if self.save_dir is not None:
            model_state_dict = {
                'subgraph_model': self.subgraph_model.state_dict(),
                'context_model': self.context_model.state_dict(),
                'subgraph_discriminator': self.subgraph_discriminator.state_dict(),
                'epoch': self.epoch,
                'loss': loss_meter.avg,
                'acc': acc_meter.avg,
            }
            torch.save(
                model_state_dict,
                os.path.join(self.save_dir, 'subgraph-%03d.pt' % self.epoch)
            )

        duration = (time.time() - start) / 60
        logging.info('* Subgraph epoch %d (%.2fmin/ep, %.2fs/batch): avg_loss=%.2f, avg_acc=%.2f'
                     % (self.epoch, duration, time_meter.avg, loss_meter.avg, acc_meter.avg))


    @torch.no_grad()
    def build_macro_graph(self, file_name=None):
        macro_data = []
        # set to evaluation mode
        self.subgraph_model.eval()

        with tqdm(total=len(self.dataset)) as p_bar:
            p_bar.set_description('Generating macro graph:')

            for i, (pairs, macro_arch_graph, cls_label, reg_label, label_class) in enumerate(self.dataset): # ！

                pairs_subgraph = pairs['subgraph'].to(self.device)
                subgraph_vec = self.subgraph_model(
                    x=pairs_subgraph.x,
                    edge_index=pairs_subgraph.edge_index,
                    batch=pairs_subgraph.batch,
                )

                macro_data.append({
                    'x': subgraph_vec.detach().clone().to('cpu'),
                    'edge_index': torch.tensor(macro_arch_graph).view(-1, 2).T.long(),
                    'cls_label': cls_label,
                    'reg_label': reg_label,
                    'label_class': label_class,
                })

                p_bar.update(1)

        if file_name is not None:
            torch.save(macro_data, file_name)

        return macro_data


class GraphTrainer(object):

    def __init__(self, model_args, dataset, lr, batch_size=2, optimizer_kwargs=None,
                 logging_interval=50, device='cuda'):
        super(GraphTrainer, self).__init__()

        self.model = GraphNetwork(
            in_channels=model_args['in_channels'],
            hidden_channels=model_args['hidden_channels'],
            num_layers=model_args['num_layers'],
            num_regs = model_args['num_regs'],
            num_classes=model_args['num_classes'], 
            **model_args.get('kwargs', {}),
        ).to(device)



        self.device = device

        # self.criterion = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_reg = torch.nn.MSELoss()
        # labels = torch.arange(batch_size)

        # set optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, **optimizer_kwargs)

        # dataset and data_loader
        self.dataset = dataset
        self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        self.data_loader_eval = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        self.epoch = 0
        self.logging_interval = logging_interval

 
    def contras_loss(self, pred, target, weights):
        if weights is not None:
            weights = weights.to(target.device)
            weights = weights[target].unsqueeze(0).T

        pred = pred / pred.norm(dim=-1, keepdim=True)
        sim = pred @ pred.T  # [N, N]

        yy = target.unsqueeze(0)
        pos_mask = yy.T == yy
        neg_mask = ~pos_mask

        # pos
        pos_logits = sim[pos_mask]
        pos_logits = torch.exp(pos_logits)
        if weights is not None:

            pos_weights = (pos_mask * weights)[pos_mask]
            pos_logits = pos_logits * pos_weights
        pos_logits = pos_logits.sum()

        # neg
        neg_logits = sim[neg_mask]
        neg_logits = torch.exp(neg_logits)
        if weights is not None:
            neg_weights = (neg_mask * weights)[neg_mask]
            neg_logits = neg_logits * neg_weights
        neg_logits = neg_logits.sum()

        loss = - torch.log(
            pos_logits / (pos_logits + neg_logits)
        )

        return loss
        
    def count_weight(self, y):
        l = y.tolist()
        uni_labels = list(set(l))
        weight = torch.tensor([l.count(uni_label)/len(l) for uni_label in uni_labels])
        return weight
    
    def step(self, logging_interval=None):
        self.epoch += 1

        loss_sum = 0.
        acc_sum = 0.
        
        self.model.set_return_prediction()
        self.model.train()
        start = time.time()

        for i, batch in enumerate(self.data_loader):

            batch_start = time.time()
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            x, cls_out, reg_out = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            
            weight = self.count_weight(batch.cls_label)
            weight = None
            loss = self.contras_loss(x, batch.cls_label, weight) + self.criterion_cls(cls_out, batch.cls_label) # + self.criterion_reg(reg_out, batch.reg_label)

            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            predicted_num_class = torch.argmax(cls_out, dim=-1)
            
            acc_sum += accuracy_score(batch.label_class.detach().clone().to('cpu'), predicted_num_class.detach().clone().to('cpu'))

        duration = (time.time() - start) / 60
        logging.info('* Entire graph  epoch %d (%.2fmin/ep): avg loss=%.2f' % (self.epoch, duration, loss_sum / (len(self.data_loader))))
        return acc_sum / (i+1)

    @torch.no_grad()
    def get_graph_embedding(self):

        self.model.set_return_embedding()
        self.model.eval()

        embeddings = []

        with tqdm(total=len(self.data_loader)) as p_bar:
            p_bar.set_description('Generating graph embedding:')

            for i, batch in enumerate(self.data_loader_eval):
                batch = batch.to(self.device)
                out = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                embeddings.append(out.detach().clone().to('cpu'))
                p_bar.update(1)

        return torch.cat(embeddings, dim=0)
