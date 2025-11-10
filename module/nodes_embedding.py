import numpy as np
import torch
import torch.nn as nn
import atom_embedding
class Nodes_Embedding(nn.Module):#node feature embedding with MLP
    def __init__(self):
        super().__init__()
        self.mol2 = '/Users/jiaoyuan/Documents/GitHub/DeepMolH/DeepMolH/module_nn/net.mol2'
        self.atom_embed = atom_embedding.Atom_Embedding(self.mol2)
        self.atom_type_vector, self.atom_part, self.atom_charge = self.atom_embed.forward()
        self.nodes_size = torch.tensor(self.atom_type_vector).shape[1]
        self.nodes_embedding = nn.Sequential(
            nn.Linear(self.nodes_size,self.nodes_size**2),
            nn.ELU(),
            nn.Linear(self.nodes_size**2,self.nodes_size**2),
            nn.ELU(),
            nn.Linear(self.nodes_size**2,self.nodes_size**2),
            nn.ELU(),
            nn.Linear(self.nodes_size**2,self.nodes_size),
        )
        
    def forward(self,mol2):
        atom_embed = atom_embedding.Atom_Embedding(mol2)
        atom_type_vector, atom_part, atom_charge = atom_embed.forward()
        atom_type_vector = torch.tensor(atom_type_vector,dtype=torch.float32)
        atom_part_embed = torch.tensor(atom_embed.atom_molecular_part(),dtype=torch.float32)
        nodes1=[]
        for atom_v,atom_part in zip(atom_type_vector,atom_part):
            atom_part_tensor = torch.full((self.nodes_size,), atom_part, dtype=torch.float32)
            atom_charge_tensor = torch.full((self.nodes_size,), self.atom_charge[atom_part], dtype=torch.float32)
            nodes1.append(self.nodes_embedding(atom_v + atom_part_tensor + atom_charge_tensor))
        return torch.stack(nodes1)
    
if __name__ == '__main__':
    mol2 = 'dataset/mol/1.mol2'
    nodes_embed = Nodes_Embedding()
    nodes_embed.forward(mol2)