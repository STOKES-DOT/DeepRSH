import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn.o3 as o3
import nodes_embedding
import bond_embedding
import math


class EGAT_layer_base(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, lmax=4):
        super().__init__()
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_heads = num_heads
        self.concat = concat
        self.lmax = lmax
        self.log_attention_weights = log_attention_weights
        self.add_skip_connection = add_skip_connection
        self.bias = bias
        if add_skip_connection:
            self.skip_linear = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.skip_linear = None
    def init_params(self):
        for tp in [self.tp_query, self.tp_key, self.tp_value, self.edge_tp_distance, self.edge_tp_bond]:
            for weight in tp.parameters():
                if weight.dim() > 1:
                    #torch.manual_seed(42) #NOTE:testing
                    nn.init.xavier_uniform_(weight)
        
        if self.skip_linear is not None:
            nn.init.xavier_uniform_(self.skip_linear.weight)
    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  
            self.attention_weights = attention_coefficients
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        if out_nodes_features.dim() != 3 or out_nodes_features.size(1) != self.num_heads:
            out_nodes_features = out_nodes_features.view(-1, self.num_heads, self.num_out_features)
        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_heads, self.num_out_features)
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class EGATlayer(EGAT_layer_base):
    def __init__(self, num_in_features, num_out_features, num_heads=3, concat=False, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False, lmax=4):
        super(EGATlayer,self).__init__(
                num_in_features, 
                num_out_features, 
                num_heads=1, 
                concat=False, 
                activation=None, dropout_prob=0.6, add_skip_connection=True, 
                bias=True, log_attention_weights=False, lmax=4
        )

        self.num_heads = num_heads
        self.concat = concat
        self.lmax = lmax
        self.sh_dim = (lmax+1)**2
        self.dropout_prob = dropout_prob
        
        #nodes-scalars
        self.irreps_nodes_in = o3.Irreps(f"{num_in_features}x0e")
        self.irreps_nodes_out = o3.Irreps(f"{num_out_features}x0e")
        #nodes_vector-atom-pairs directions
        
        self.irreps_dir = o3.Irreps.spherical_harmonics(lmax=lmax)
        #equivalent K, Q, V
        self.tp_query = o3.FullyConnectedTensorProduct(
            self.irreps_nodes_in,
            self.irreps_dir,
            
            o3.Irreps(f"1x0e")
        )
        self.tp_key = o3.FullyConnectedTensorProduct(
            self.irreps_nodes_in,
            self.irreps_dir,
            o3.Irreps(f"1x0e")
        )
        
        self.tp_value = o3.FullyConnectedTensorProduct(
            self.irreps_nodes_in,
            self.irreps_dir,
            o3.Irreps(f"{num_out_features}x0e")
        )
        self.tp_queries = nn.ModuleList([self.tp_query for _ in range(self.num_heads)])
        self.tp_keys = nn.ModuleList([self.tp_key for _ in range(self.num_heads)])
        self.tp_values = nn.ModuleList([self.tp_value for _ in range(self.num_heads)])
        #edges equivariant embedding
        self.egde_tp_gaussian = o3.FullyConnectedTensorProduct(
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e')
        )
        self.edge_tp_bond = o3.FullyConnectedTensorProduct(
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e')
        )
                
        self.edge_tp_bond_contribution = o3.FullyConnectedTensorProduct(
            o3.Irreps("1x0e"),
            o3.Irreps('1x0e'),
            o3.Irreps('1x0e')
        )
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
    def init_params(self):

        for head in range(self.num_heads):
            for param in self.tp_queries[head].parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            for param in self.tp_keys[head].parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            for param in self.tp_values[head].parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        for param in self.edge_tp_bond.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        for param in self.edge_tp_gaussian.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
        super().init_params()
    def forward(self, node_features, edges_features_distance, edges_features_bond, edge_direction, degree_matrix, cut_off=5.0):
        num_nodes = node_features.shape[0]
        num_edges = edge_direction.shape[0]
        sh_features = o3.spherical_harmonics(self.irreps_dir, edge_direction, normalize=True, normalization='component')
        sh_features = sh_features.reshape(num_nodes,num_nodes, (self.lmax+1)**2)
        #mask for disconnected nodes
        connectivity_mask = torch.where(degree_matrix>0, 0.0, -1e9)
        #node features projection
        nodes_i = node_features.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        nodes_j = node_features.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        #bond type embedding
        edges_features_bond = edges_features_bond.unsqueeze(-1).reshape(-1, 1)
        edge_bond_contribution = self.edge_tp_bond(
                edges_features_bond.reshape(-1, edges_features_bond.shape[-1]), 
                torch.ones(edges_features_bond.numel() // edges_features_bond.shape[-1], 1, device=edges_features_bond.device)
            ).reshape(num_nodes, num_nodes, 1)
        edge_gaussian_contribution = self.egde_tp_gaussian(
            edges_features_distance.reshape(-1, 1), 
            torch.ones(edges_features_distance.numel(), 1, device=edges_features_distance.device)
        ).reshape(num_nodes, num_nodes, 1)
        
        head_outs = []
        attention_weights_list = []
        
        for head in range(self.num_heads):
            query_input = nodes_i.reshape(-1, self.num_in_features)
            query = self.tp_queries[head](query_input, sh_features.reshape(-1,self.sh_dim))
            query = query.reshape(num_nodes, num_nodes, 1)
            
            key_input = nodes_j.reshape(-1, self.num_in_features)
            key = self.tp_keys[head](key_input, sh_features.reshape(-1,self.sh_dim))
            key = key.reshape(num_nodes, num_nodes, 1)
            
            raw_attention = (query*key.transpose(0,1)) / math.sqrt(self.num_heads)
            raw_attention += edge_bond_contribution
            raw_attention += edge_gaussian_contribution
            
            attention_scores = self.leakyReLU(raw_attention)+connectivity_mask.unsqueeze(-1)
            attention_weight = F.softmax(attention_scores, dim=1)
            attention_weight = F.dropout(attention_weight)
            attention_weights_list.append(attention_weight)
            
            value_input = nodes_j.reshape(-1, self.num_in_features)
            value = self.tp_values[head](value_input, sh_features.reshape(-1,self.sh_dim))
            value = value.reshape(num_nodes, num_nodes, self.num_out_features)
            
            aggregated = torch.matmul(torch.Tensor(attention_weight.transpose(2,1)), value)
            aggregated = aggregated.squeeze(1)
            head_outs.append(aggregated)

        out_nodes_features = torch.cat(head_outs, dim=-1)
        attention_cof = torch.cat(attention_weights_list, dim=-1)
        
        updated_nodes = self.skip_concat_bias(attention_cof, node_features, out_nodes_features)
        with torch.no_grad():
            node_similarity = torch.matmul(updated_nodes, updated_nodes.T)
            distance_decay = -edges_features_distance.squeeze(-1)
            updated_connectivity = torch.sigmoid(node_similarity) * distance_decay * degree_matrix
            
            if updated_connectivity.numel() > 0:
                updated_connectivity = F.layer_norm(
                    updated_connectivity, 
                    updated_connectivity.shape[-1:]
                )
            updated_connectivity = updated_connectivity + updated_connectivity.T
        return updated_nodes, updated_connectivity

