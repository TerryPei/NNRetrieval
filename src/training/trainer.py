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

            # print('subgraph_vec:', subgraph_vec.size())
            # print('pos_context_vec:', pos_context_vec.size())
            # print('neg_context_vec:', neg_context_vec.size())

            out_pos = self.subgraph_discriminator(subgraph_vec=subgraph_vec, context_vec=pos_context_vec)
            out_neg = self.subgraph_discriminator(subgraph_vec=subgraph_vec, context_vec=neg_context_vec)

            # print('out_pos:', out_pos.size())
            # print('out_neg:', out_neg.size())

            # if self.epoch == 10:
            # logging.debug('pos:', out_pos[0])
            # logging.debug('neg:', out_neg[0])

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
                # print(pairs.keys())# dict_keys(['subgraph', 'pos_context', 'neg_context'])
                pairs_subgraph = pairs['subgraph'].to(self.device)
                subgraph_vec = self.subgraph_model(
                    x=pairs_subgraph.x,
                    edge_index=pairs_subgraph.edge_index,
                    batch=pairs_subgraph.batch,
                )
                # print(subgraph_vec.shape) # [16, 256]
                # exit(0)
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

        # self.model = MLP(input_dim=model_args['in_channels'], output_dim=model_args['num_classes'])

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

    # def contras_loss(self, x, cls_label):
    #     # x: [N, D]
    #     # cls_label: [N,] \in [0, C)
    #     # print(x.shape, cls_label.shape, torch.unique(cls_label))
    #     assert x.shape[0] == cls_label.shape[0]
    #     batch_size = x.shape[0]

    #     # x = F.normalize(x, dim=-1)
    #     x = x / x.norm(dim=-1, keepdim=True)

    #     sim = x @ x.T #[N, N]

    #     # print(sim[sim < 0.])
    #     # sim(i, j) = x_i @ x_j is positive when label[i]==label[j]
    #     mask = torch.zeros((batch_size, batch_size)).bool()
    #     # print(mask.dtype)
    #     for i in range(batch_size):
    #         j = i
    #         for j in range(batch_size):
    #             if cls_label[i] == cls_label[j]:
    #                 mask[i, j] = True
    #             else:
    #                 mask[i, j] = False

    #     pos_logits = sim[mask]
    #     pos_logits = torch.exp(pos_logits)
    #     neg_logits = sim[~mask]
    #     neg_logits = torch.exp(neg_logits)

    #     loss = - torch.log(pos_logits.sum() / (pos_logits.sum() + neg_logits.sum()))

    #     return loss
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
            # print(weights.shape)
            # print(weights[0])
            # print(pos_mask)
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
# class SimCLRTrainer(object):
    
#     def __init__(self, model_args, dataset, lr, batch_size=2, optimizer_kwargs=None,
#                  logging_interval=50, device='cuda'):
#         super(GraphTrainer, self).__init__()

#         self.model = GraphNetwork(
#             in_channels=model_args['in_channels'],
#             hidden_channels=model_args['hidden_channels'],
#             num_layers=model_args['num_layers'],
#             **model_args.get('kwargs', {}),
#         ).to(device)

#         self.device = device

#         self.criterion = torch.nn.CrossEntropyLoss()

#         # set optimizer
#         if optimizer_kwargs is None:
#             optimizer_kwargs = {}
#         self.optimizer = optim.Adam(self.model.parameters(),
#                                     lr=lr, **optimizer_kwargs)

#         # dataset and data_loader
#         self.dataset = dataset
#         self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#         self.epoch = 0
#         self.logging_interval = logging_interval
    
#     def info_nce_loss(self, features, n_views=2):
#         batch_size = features.shape[0]
#         # labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
#         # labels = torch.eyes(batch_size)

#     def step(self, logging_interval=None):
#         self.epoch += 1
#         loss_sum = 0.
#         i = 0

#         logging_interval = self.logging_interval if logging_interval is None else logging_interval

#         self.model.set_return_prediction()
#         self.model.train()

#         start = time.time()

#         acc_sum = 0.
#         for i, batch in enumerate(self.data_loader):

#             batch_start = time.time()
#             batch = batch.to(self.device)

#             self.optimizer.zero_grad()

#             # out = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)

#             emb = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)

#             loss = 0.9 * self.criterion_cls(cls_out, batch.cls_label) + 0.1 * self.criterion_reg(reg_out, batch.reg_label)
#             # + constrative(x, )
#             loss.backward()
#             self.optimizer.step()
#             loss_sum += loss.item()
#             predicted_num_class = torch.argmax(cls_out, dim=-1)
#             # print(batch.label_class.shape,  predicted_num_class.shape, torch.unique(batch.label_class), torch.unique(predicted_num_class))
#             # exit(0)

#             acc_sum += accuracy_score(batch.label_class, predicted_num_class)
#             # 
#             # logging
#             # if i % logging_interval == 0:
#             #     batch_duration = (time.time() - batch_start)
#             #     logging.info(
#             #         '  epoch %d batch %d (%.2fs/batch): loss=%.2f (%.2f)'
#             #         % (self.epoch, i, batch_duration, loss.item(), loss_sum / (i + 1))
#             #     )
#         duration = (time.time() - start) / 60
#         logging.info('* Entire graph  epoch %d (%.2fmin/ep): avg loss=%.2f avg acc=%.2f' % (self.epoch, duration, loss_sum / (i + 1), acc_sum / (i+1)))

#         return acc_sum / (i+1)

#     @torch.no_grad()
#     def get_graph_embedding(self):

#         self.model.set_return_embedding()
#         self.model.eval()

#         embeddings = []

#         with tqdm(total=len(self.data_loader)) as p_bar:
#             p_bar.set_description('Generating graph embedding:')

#             for i, batch in enumerate(self.data_loader):
#                 batch = batch.to(self.device)
#                 out = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
#                 # print(out)
#                 # exit(0)
#                 embeddings.append(out.detach().clone().to('cpu'))
#                 p_bar.update(1)

#         return torch.cat(embeddings, dim=0)



# class SimCLR(object):
    
#     def __init__(self, args, **kwargs):
#         self.args = args
#         self.model = torch.load(args.model_path)['encoder'].to(self.args.device)
#         self.optimizer = kwargs['optimizer']
#         self.scheduler = kwargs['scheduler']
#         # self.writer = SummaryWriter()
#         # logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
#         self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

#     def info_nce_loss(self, features):

#         labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(self.args.device)

#         features = F.normalize(features, dim=1)

#         similarity_matrix = torch.matmul(features, features.T)
#         # assert similarity_matrix.shape == (
#         #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
#         # assert similarity_matrix.shape == labels.shape

#         # discard the main diagonal from both: labels and similarities matrix
#         mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
#         labels = labels[~mask].view(labels.shape[0], -1)
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#         # assert similarity_matrix.shape == labels.shape

#         # select and combine multiple positives
#         positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

#         # select only the negatives the negatives
#         negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

#         logits = torch.cat([positives, negatives], dim=1)
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

#         logits = logits / self.args.temperature
#         return logits, labels

#     def train(self, train_loader):

#         scaler = GradScaler(enabled=self.args.fp16_precision)

#         # save config file
#         save_config_file(self.writer.log_dir, self.args)

#         n_iter = 0
#         logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
#         logging.info(f"Training with gpu: {self.args.disable_cuda}.")

#         for epoch_counter in range(self.args.epochs):
#             for images, _ in tqdm(train_loader):
#                 images = torch.cat(images, dim=0)

#                 images = images.to(self.args.device)

#                 with autocast(enabled=self.args.fp16_precision):
#                     features = self.model(images)
#                     logits, labels = self.info_nce_loss(features)
#                     loss = self.criterion(logits, labels)

#                 self.optimizer.zero_grad()

#                 scaler.scale(loss).backward()

#                 scaler.step(self.optimizer)
#                 scaler.update()

#                 if n_iter % self.args.log_every_n_steps == 0:
#                     top1, top5 = accuracy(logits, labels, topk=(1, 5))
#                 n_iter += 1

#             # warmup for the first 10 epochs
#             if epoch_counter >= 10:
#                 self.scheduler.step()
#             logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

#         logging.info("Training has finished.")
#         # save model checkpoints
#         checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
#         save_checkpoint({
#             'epoch': self.args.epochs,
#             'arch': self.args.arch,
#             'state_dict': self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#         }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
#         logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")