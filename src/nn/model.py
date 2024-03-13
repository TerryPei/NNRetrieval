import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphNetwork(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers, num_regs=None, num_classes=None,
                 gnn_operator=GCNConv, drop_rate=0., **kwargs):
        super(GraphNetwork, self).__init__()
        self.drop_rate = drop_rate
        encoder = [
            gnn_operator(in_channels=in_channels, out_channels=hidden_channels, **kwargs)
        ]
        if num_layers > 1:
            for i in range(num_layers - 1):
                encoder.append(
                    gnn_operator(in_channels=hidden_channels, out_channels=hidden_channels, **kwargs)
                )
        self.encoder = nn.ModuleList(encoder)

        self.classifer = nn.Linear(in_features=hidden_channels, out_features=num_classes) if num_classes is not None else None
        #[n, d] -> [n, num_classes]
        self.reg = nn.Linear(in_features=hidden_channels, out_features=num_regs) if num_regs is not None else None
        #[n, d] -> [n, num_reg]
        self._return_embedding = False

        # self.fc = nn.Linear(in_features=hidden_channels, out_features=hidden_channels)

    def set_return_embedding(self, return_embedding: bool = True):
        self._return_embedding = return_embedding

    def set_return_prediction(self):
        self.set_return_embedding(return_embedding=False)

    def forward(self, x, edge_index, batch, edge_weight=None):
        # x = x[:2]
        # print(x.shape) #[182485, 38] 182485 is dynamic
        # exit(0)
        if self.drop_rate != 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        
        x = self.encoder[0](x=x, edge_index=edge_index, edge_weight=edge_weight) # !
        ################################################################
        for layer in self.encoder[1:]:
            x = F.elu(x)
            if self.drop_rate != 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = layer(x=x, edge_index=edge_index, edge_weight=edge_weight)

        if batch is None:
            batch = torch.zeros(x.size(0)).to(x.device)
        
        x = global_mean_pool(x=x, batch=batch)
        # print(x.shape) # [5380, 256]
        # # exit(0)
        # print(batch.cls_label.shape) # [5380, 256]
        # exit(0)
        ################################################################

        if self._return_embedding:
            return x
        
        # cls_out, reg_out = None, None
        if self.classifer is None and self.reg is None:
            return x

        if self.classifer is not None:
            cls_out = self.classifer(x)

        if self.reg is not None:
            reg_out = self.reg(x)
        
        return x, cls_out, reg_out
        # if self._return_embedding:
        #     return x
        # else:
        #     # print(self.classifer == None, self.reg == None)
        #     # return self.classifer(x) if self.classifer is not None else x
        #     return self.classifer(x), self.reg(x) if ((self.classifer is not None) and (self.reg is not None)) else x、
class SubgraphDiscriminator(nn.Module):

    def __init__(self, embed_dim):
        super(SubgraphDiscriminator, self).__init__()
        self.fc = nn.Linear(in_features=embed_dim, out_features=1)

    def forward(self, subgraph_vec, context_vec):
        return torch.sigmoid(
            self.fc(
                torch.cat([subgraph_vec, context_vec], dim=-1)
            )
        )

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_array = [256, 256], non_linear_function_array=[nn.ReLU, nn.ReLU]):
        super().__init__()
        self.linear_functions = []
        self.non_linear_functions = [x() for x in non_linear_function_array]
        self.hidden_layers = len(hidden_dim_array)
        for l in range(self.hidden_layers):
            self.linear_functions.append(nn.Linear(input_dim, hidden_dim_array[l]))
            input_dim = hidden_dim_array[l]
        self.linear_functions = nn.ModuleList(self.linear_functions)
        self.final_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = x
        for i in range(self.hidden_layers):
            out = self.linear_functions[i](out)
            out = self.non_linear_functions[i](out)
        out = self.final_linear(out)
        return out
    

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim_array = [256, 128], non_linear_function_array=[nn.ReLU, nn.ReLU], drop_rate=0., **kwargs):
        super().__init__()
        self.encoder = MLP(input_dim, emb_dim, hidden_dim_array = hidden_dim_array, non_linear_function_array=non_linear_function_array)
        self.decoder = MLP(emb_dim, input_dim, hidden_dim_array = list(reversed(hidden_dim_array)), non_linear_function_array=list(reversed(non_linear_function_array)))
        self._return_embedding = False

    def set_return_embedding(self, return_embedding: bool = True):
        self._return_embedding = return_embedding

    def forward(self, x): # graph -> [N, K]
        emb = self.encoder(x)
        if self._return_embedding:
            return emb
        x = self.decoder(emb)
        return x

# class FineTuningClassifier(nn.Module):
#     def __init__(self, in_features, hidden_features, num_layers, num_classes=None,
#                 hidden_op=nn.Linear, drop_rate=0., **kwargs):
#         super(FineTuningClassifier, self).__init__()
#         self.drop_rate = drop_rate
#         encoder = [
#             hidden_op(in_features=in_features, out_features=hidden_features)
#         ]
#         if num_layers > 1:
#             for i in range(num_layers - 1):
#                 encoder.append(
#                     hidden_op(in_features=in_features, out_features=hidden_features)
#                 )
#         self.encoder = nn.ModuleList(encoder)
#         self.head = nn.Linear(in_features=in_features, out_features=num_classes)

#     def forward(self, x):
#             # x = x[:2]
#         if self.drop_rate != 0.:
#             x = F.dropout(x, p=self.drop_rate, training=self.training)
#         x = self.encoder(x)
#         cls_out = self.head(x)
#         return cls_out


