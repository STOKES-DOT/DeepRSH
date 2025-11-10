import torch
import torch.nn as nn
import torch.nn.functional as F
import pyscf
from pyscf import gto, scf, dft
import numpy as np

import os

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        
    def _mol2_to_xyz(self,mol2):
        xyz_lines = []
        with open(mol2, 'r') as f:
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
                    atom_symbol = parts[1][:2].title()
                    x, y, z = parts[2:5] 
                    xyz_lines.append(f"{atom_symbol} {x} {y} {z}")
        return xyz_lines
    def calculate_energy(self, charge, spin, xc, mol2, basis = 'sto-3g'):
        mol_xyz = self._mol2_to_xyz(mol2)
        mol = gto.M(
            atom=mol_xyz,
            basis=basis,
            charge=charge,
            spin=spin,
        )
        xcf = xc
        if charge == 0 and spin == 0:
            mf = scf.RKS(mol)
            mf.xc = xcf
            mf.verbose = 0
            mf.kernel()
            nocc = mol.nelectron // 2
            mo_energy_alpha = mf.mo_energy
            homo = mo_energy_alpha[nocc-1]
            lumo = mo_energy_alpha[nocc]
        else:
            mf = scf.UKS(mol)
            mf.xc = xcf
            mf.verbose = 0
            mf.kernel()
            mo_energy_alpha = mf.mo_energy[0]
            mo_energy_beta = mf.mo_energy[1]
            nocc_alpha = (mol.nelectron + mol.spin) // 2
            nocc_beta = (mol.nelectron - mol.spin) // 2
            
            homo_alpha = mo_energy_alpha[nocc_alpha-1]
            lumo_alpha = mo_energy_alpha[nocc_alpha]
            homo_beta = mo_energy_beta[nocc_beta-1]
            lumo_beta = mo_energy_beta[nocc_beta]
            
            homo = max(homo_alpha, homo_beta)
            lumo = min(lumo_alpha, lumo_beta)
            
        return mf.e_tot, homo, lumo
    
    def get_j_function(self,xc,mol2,basis = 'sto-3g'):

        E_N, E_homo, E_lumo = self.calculate_energy(charge=0, spin=0,xc=xc,mol2=mol2,basis=basis)
        E_N_plus_1, E_homo_plus_1, E_lumo_plus_1 = self.calculate_energy(charge=-1, spin=1,xc=xc,mol2=mol2,basis=basis)
        E_N_minus_1, E_homo_minus_1, E_lumo_minus_1 = self.calculate_energy(charge=1, spin=1,xc=xc,mol2=mol2,basis=basis)
        
        J_N = abs(E_homo + E_N_minus_1 - E_N)
        J_N_plus_1 = abs(E_homo_plus_1 + E_N - E_N_plus_1)
        J = J_N**2 + J_N_plus_1**2
        return J
    
    def forward(self,xc,mol2,basis = 'sto-3g'):
        J = self.get_j_function(xc,mol2,basis)
        return J
