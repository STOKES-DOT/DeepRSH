import torch
import torch.nn as nn
import torch.nn.functional as F
import nodes_embedding
import bond_embedding


class GATlayer_Base(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_heads=2, concat=False, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        
        self.head_projections = nn.ModuleList([
            nn.Linear(num_in_features, num_out_features, bias=False)
            for _ in range(num_of_heads)
        ])
        
        self.scoring_fn_target = nn.Parameter(torch.Tensor(num_of_heads, num_out_features, 1))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(num_of_heads, num_out_features, 1))
        self.edge_distance_proj = nn.Linear(1, num_of_heads, bias=False)
        self.edge_bond_proj = nn.Linear(1, num_of_heads, bias=False)
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2) 
        self.softmax = nn.Softmax(dim=-1)  
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  
        self.attention_weights = None 


    def init_params(self):

        nn.init.xavier_uniform_(self.proj_param)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  
            self.attention_weights = attention_coefficients
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        if out_nodes_features.dim() != 3 or out_nodes_features.size(1) != self.num_of_heads:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads, self.num_out_features)
        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)




class GATlayer(GATlayer_Base):
    def __init__(self, num_in_features, num_out_features, num_heads=1, dropout=0.5, concat=False,activation=nn.ELU(), add_skip_connection=None, bias=None, log_attention_weights=True,esp=1e-6):
        super(GATlayer, self).__init__(num_in_features, num_out_features, num_heads, concat=concat, activation=activation, add_skip_connection=add_skip_connection, bias=bias, log_attention_weights=log_attention_weights)
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.activation = activation
        self.add_skip_connection = add_skip_connection
        self.bias = bias
        self.log_attention_weights = log_attention_weights
        self.esp = esp
        
    def forward(self,nodes_features,degree_matrix,edges_features_distance,edges_features_bond,cutoff=0):
        connectivity_mask = torch.where(degree_matrix>0, 0.0, -1e9)
    
        num_of_nodes = nodes_features.size(0)
        assert connectivity_mask.shape == (num_of_nodes,num_of_nodes),f"connectivity_mask shape error,expected {(num_of_nodes,num_of_nodes)},got {connectivity_mask.shape}"
        nodes_features_proj = nodes_features.unsqueeze(0)
        head_outs = []
        for head, proj_layer in enumerate(self.head_projections):
            head_features = proj_layer(nodes_features)  # (num_nodes, num_out_features)
            head_outs.append(head_features)
        nodes_features_proj = torch.stack(head_outs, dim=0)
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = torch.bmm(nodes_features_proj,self.scoring_fn_source)#Key
        scores_target = torch.bmm(nodes_features_proj,self.scoring_fn_target)#Query

        #compute attention coefficients
        all_scores = self.leakyReLU(scores_source + scores_target.transpose(1,2))
        
    
        edge_distance_contribution = self.edge_distance_proj(-edges_features_distance.unsqueeze(-1))
        edge_distance_contribution = edge_distance_contribution.reshape(self.num_of_heads, num_of_nodes, num_of_nodes)
        all_scores += edge_distance_contribution

        
        edge_bond_contribution = self.edge_bond_proj(edges_features_bond.unsqueeze(-1))
        edge_bond_contribution = edge_bond_contribution.reshape(self.num_of_heads, num_of_nodes, num_of_nodes)
        all_scores += edge_bond_contribution
        try:
            all_attention_coefficients = torch.softmax(all_scores + connectivity_mask.unsqueeze(0), dim=-1)
        except RuntimeError:
            print("Softmax numerical instability, using fallback")
            max_vals = torch.max(all_scores + connectivity_mask.unsqueeze(0), dim=-1, keepdim=True)[0]
            stable_scores = (all_scores + connectivity_mask.unsqueeze(0)) - max_vals
            all_attention_coefficients = torch.softmax(stable_scores, dim=-1)
        
        out_nodes_features = torch.bmm(all_attention_coefficients,nodes_features_proj)
        out_nodes_features = out_nodes_features.transpose(2,1)
        
        updated_nodes_features = self.skip_concat_bias(all_attention_coefficients, nodes_features, out_nodes_features)
        with torch.no_grad():
            node_similarity = torch.matmul(updated_nodes_features, updated_nodes_features.T)
            distance_decay = -edges_features_distance.squeeze(-1)
            updated_connectivity_mask = torch.sigmoid(node_similarity)*distance_decay*(degree_matrix>0).float()
            if updated_connectivity_mask.numel() > 0:
                updated_connectivity_mask = updated_connectivity_mask+self.esp
                updated_connectivity_mask = F.layer_norm(
                    updated_connectivity_mask, 
                    updated_connectivity_mask.shape[-1:]
                )
            updated_connectivity_mask = updated_connectivity_mask + updated_connectivity_mask.T

        return (updated_nodes_features, updated_connectivity_mask)

