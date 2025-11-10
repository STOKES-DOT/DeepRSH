import torch 
import torch.nn as nn
import numpy as np

class Bond_Embedding(nn.Module):
    def __init__(self, mol2,files=None,index=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mol2 = mol2
        self.bond_type_matrix = self.get_bond_type()
        self.gb_matrix = self.gaussian_basis_matrix()
        self.files = files
        self.index = index
    def get_atom_pairs_distance(self):
        atoms = []
        with open(self.mol2, 'r') as f:
            lines = f.readlines()
        atom_section = False
        for line in lines:
            if line.startswith('@<TRIPOS>ATOM'):
                atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                atom_section = False
                continue
            if atom_section and line.strip():
                parts = line.split()
                if len(parts) >= 6: 
                    atom_index = int(parts[0])
                    atom_name = parts[1][:2].title()
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    atoms.append((atom_index,atom_name,x,y,z))
        distance = []
        index = []
        n_atoms = len(atoms)
        for i in range(n_atoms):
            for j in range(n_atoms):
                index1, atom1, x1, y1, z1 = atoms[i]
                index2, atom2, x2, y2, z2 = atoms[j]
                distance.append((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
                index.append((index1,index2))
        return distance, index
    def get_atom_pairs_direction(self):
        atoms = []
        with open(self.mol2, 'r') as f:
            lines = f.readlines()
        atom_section = False
        for line in lines:
            if line.startswith('@<TRIPOS>ATOM'):
                atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                atom_section = False
                continue
            if atom_section and line.strip():
                parts = line.split()
                if len(parts) >= 6: 
                    atom_index = int(parts[0])
                    atom_name = parts[1][:2].title()
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    atoms.append((atom_index,atom_name,x,y,z))
        direction = []
        index = []
        n_atoms = len(atoms)
        for i in range(n_atoms):
            for j in range(n_atoms):
                index1, atom1, x1, y1, z1 = atoms[i]
                index2, atom2, x2, y2, z2 = atoms[j]
                direction.append((x2-x1,y2-y1,z2-z1))
                index.append((index1,index2))
        return direction
    def _gaussian_basis(self,rij,sigma=20,mu=3,cutoff=20):
        if np.sqrt(rij) <= cutoff:
            var_eps = 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(np.sqrt(rij)-mu)**2/(2*sigma**2))
        else:
            var_eps = 0
        
        return var_eps
    def gaussian_basis_matrix(self,sigma=1,mu=0):
        distance, index = self.get_atom_pairs_distance()
        max_index = 0
        for idx_pair in index:
            max_index = max(max_index,max(idx_pair))
        gb_matrix = np.zeros((max_index, max_index))
        for dist, (i,j) in zip(distance,index):
            gb_value = self._gaussian_basis(dist,sigma,mu)
            gb_matrix[i-1,j-1] = gb_value
            gb_matrix[j-1,i-1] = gb_value
        return gb_matrix

    def get_bond_type(self):
        bond_type = []
        atom1 = []
        atom2 = []
        with open(self.mol2, 'r') as f:
            lines = f.readlines()
        atom_section = False
        for line in lines:
            if line.startswith('@<TRIPOS>BOND'):
                atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                atom_section = False
                continue
            if atom_section and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    bond_type.append(str(parts[3]))
                    atom1.append(int(parts[1]))
                    atom2.append(int(parts[2]))
        for i in range(len(atom1)):
            if bond_type[i] == '1':
                bond_type[i] = 1
            if bond_type[i] == '2':
                bond_type[i] = 2
            if bond_type[i] == '3':
                bond_type[i] = 3
            if bond_type[i] == 'ar':
                bond_type[i] = 1.5
            if bond_type[i] == 'am':
                bond_type[i] = 1.2
        max_index = max(max(atom1),max(atom2))
        bond_type_matrix = np.zeros((max_index, max_index))
        for values, (i,j) in zip(bond_type,zip(atom1,atom2)):
            bond_type_matrix[i-1,j-1] = values
            bond_type_matrix[j-1,i-1] = values
        return bond_type_matrix
    def save_matrix(self):
        from scipy.sparse import csr_matrix, save_npz
        gaussian_matrix = csr_matrix(self.gb_matrix)
        save_npz(f'{self.files}/{self.index}_g.npz',gaussian_matrix)
        bond_type_matrix = csr_matrix(self.bond_type_matrix)
        save_npz(f'{self.files}/{self.index}_b.npz',bond_type_matrix)
    def get_degree_matrix(self):
        bond_type_matrix = self.get_bond_type()
        degree_matrix = np.zeros_like(bond_type_matrix)
        num_atoms = bond_type_matrix.shape[0]
        for i in range(num_atoms):
            for j in range(num_atoms):
                if bond_type_matrix[i,j] != 0:
                    degree_matrix[i,j] = 1
        return degree_matrix
    def forward(self):
        distance, index = self.get_atom_pairs_distance()
        direction= self.get_atom_pairs_direction()
        bond_type_matrix = self.get_bond_type()
        gb_matrix = self.gaussian_basis_matrix()
        degree_matrix = self.get_degree_matrix()
        return  gb_matrix, bond_type_matrix, degree_matrix, direction