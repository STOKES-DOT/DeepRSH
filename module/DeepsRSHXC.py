import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, MessagePassing
from torch_geometric.data import Data, Batch
from decimal import Decimal
import atom_embedding
import bond_embedding
import nodes_embedding
import GAT_Layer


class MolEmbeddingLayer(nn.Module):
    def __init__(self):
        super(MolEmbeddingLayer, self).__init__()
        self.nodes = nodes_embedding.Nodes_Embedding()
        
    def forward(self, mol2):
        self.edges = bond_embedding.Bond_Embedding(mol2)
        node_feat = self.nodes.forward(mol2)
        edge_feat_dis, edge_feat_bond, degree_tensor, edges_direction = self.edges.forward()
        return node_feat, edge_feat_dis, edge_feat_bond, degree_tensor, edges_direction
    
class GNNLayer(nn.Module):
    def __init__(self, node_in_dim=128, num_out_dim=128, num_heads=8, num_gat_layers=2,num_egat_layers=2, dropout=0.6):
        super(GNNLayer, self).__init__()
        self.num_heads = num_heads
        self.num_egat_layers = num_egat_layers
        self.num_gat_layers = num_gat_layers
        self.num_out_dim = num_out_dim
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            self.gnn_layers.append(
                GAT_Layer.GATlayer(
                    num_in_features=node_in_dim,
                    num_out_features=num_out_dim,
                    num_heads=num_heads,
                    concat=False,
                    activation=nn.ELU() if i < num_gat_layers - 1 else None,
                    add_skip_connection=True
                )
            )
    def forward(self, node_feat, edges1, edges2, edges_direction, degree_tensor, cut_off=5.0):
        for i,layer in enumerate(self.gnn_layers):
            if i < self.num_gat_layers:
                node_feat, connectivity_mask = layer.forward(node_feat, degree_tensor, edges1, edges2,  cut_off)
            else:
                node_feat, connectivity_mask = layer.forward(node_feat, edges1, edges2, edges_direction, degree_tensor, cut_off)
            if i < self.num_egat_layers + self.num_gat_layers - 1:
                node_feat = F.dropout(node_feat, p=self.dropout, training=self.training)
        return node_feat, connectivity_mask
    
class GatedAttentionPooling(nn.Module):
    def __init__(self, node_dim, graph_dim):
        super(GatedAttentionPooling, self).__init__()
        self.global_attention = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(node_dim, graph_dim),
                nn.Sigmoid()
            ),
            nn = None
        )
        self.projection = nn.Linear(node_dim, graph_dim)
    def forward(self, x, batch):
        graph_representation = self.global_attention(x, batch)
        graph_representation = self.projection(graph_representation)
        return graph_representation
class ParameterNN(nn.Module):
    def __init__(self, node_in_dim, hidden_size=128, output_size=3):
        super(ParameterNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(node_in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)
    
class DeepsRSHXC(nn.Module):
    def __init__(self, num_heads=8, num_gat_layers=2, dropout=0.6):
        super(DeepsRSHXC, self).__init__()
        self.mol2 = f'/Users/jiaoyuan/Documents/GitHub/ADOPTXC/module/net.mol2'
        self.nodes_feat, self.edges_feat_dis,self.edges_feat_bond, self.degree_tensor, self.edges_direction = MolEmbeddingLayer().forward(self.mol2)
        self.gnn_layer = GNNLayer(
            node_in_dim=self.nodes_feat.shape[1], 
            num_out_dim=self.nodes_feat.shape[1], 
            num_heads=num_heads, 
            num_gat_layers=num_gat_layers, 
            dropout=dropout)
        self.gated_attention_pooling = GatedAttentionPooling(
            node_dim=self.nodes_feat.shape[1],
            graph_dim=self.nodes_feat.shape[1])
        self.parameter = ParameterNN(
            node_in_dim=self.nodes_feat.shape[1],
            output_size=3)
    @staticmethod
    def create_tunable_b3lyp(xc_alpha=0.2, xc_beta=0.72, xc_omega=None, lda_frac=0.08, lyp_frac=0.81, vwn_frac=0.19):
    # Verify exchange part sums to 1
        total_x = xc_alpha + xc_beta + lda_frac
        if xc_omega is None:
        # Standard B3LYP form, no range separation
            xc_str = f'HF*{Decimal(str(xc_alpha))} + LDA*{Decimal(str(lda_frac))} + B88*{Decimal(str(xc_beta))}, LYP*{Decimal(str(lyp_frac))} + VWN*{Decimal(str(vwn_frac))}'
        else:
        # Range separated version
            xc_str = f'RSH({Decimal(str(xc_omega))},{Decimal(str(xc_alpha))},-{Decimal(str(xc_beta))}) + LDA*{Decimal(str(lda_frac))}, LYP*{Decimal(str(lyp_frac))} + VWN*{Decimal(str(vwn_frac))}'
        return xc_str
    def forward(self,mol2):
        nodes_feat, edges_feat_dis, edges_feat_bond, degree_tensor, edges_direction = MolEmbeddingLayer().forward(mol2)
        degree_tensor = torch.tensor(degree_tensor, dtype=torch.float32)
        edges_feat_dis = torch.tensor(edges_feat_dis, dtype=torch.float32)
        edges_feat_bond = torch.tensor(edges_feat_bond, dtype=torch.float32)
        edges_direction = torch.tensor(edges_direction, dtype=torch.float32)
        out_nodes_features, connectivity_mask = self.gnn_layer.forward(nodes_feat, edges_feat_dis, edges_feat_bond, edges_direction,degree_tensor)
        graph_representation = self.gated_attention_pooling.forward(out_nodes_features, batch=torch.zeros(out_nodes_features.size(0), dtype=torch.long))
        parameters = self.parameter.forward(graph_representation)

        xc_functional = self.create_tunable_b3lyp(xc_alpha=float(parameters[0][0]), xc_beta=float(parameters[0][1]), xc_omega=float(parameters[0][2]))
        return xc_functional

if __name__ == '__main__':
    deepxc = DeepsRSHXC()
    mol2 = f'/Users/jiaoyuan/Documents/GitHub/ADOPTXC/module/net.mol2'
    parameters = deepxc.forward(mol2)
    print(parameters)