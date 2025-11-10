import pyscfad as pyscf
from pyscfad import gto, scf, dft
from pyscfad.dft import numint
import numpy as np
from pyscfad import lib
from decimal import Decimal
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
class EnergyGetter():
    def __init__(self, mol2, alpha=0.5, beta=0.5, omega=0.1, basis='sto-3g'):
        super(EnergyGetter, self).__init__()
        self.mol2 = mol2
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.xyz = self._mol2_to_xyz()
        self.basis = basis
        
    def _mol2_to_xyz(self):
        xyz_lines = []
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
                    atom_symbol = parts[1][:2].title()
                    x, y, z = parts[2:5] 
                    xyz_lines.append(f"{atom_symbol} {x} {y} {z}")
        return xyz_lines
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
    def calculate_energy(self, charge, spin, alpha=None, beta=None, omega=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if omega is None:
            omega = self.omega
            
        mol = gto.Mole(
            atom=self.xyz,
            basis=self.basis,
            charge=charge,
            spin=spin,
        )
        
        xc_functional = self.create_tunable_b3lyp(xc_alpha=self.alpha, xc_beta=self.beta, xc_omega=self.omega)
        print(alpha, beta, omega)
        if charge == 0 and spin == 0:
            mf = dft.RKS(mol)
            mf.xc = xc_functional
            mf.verbose = 0
            mf.kernel()
            nocc = mol.nelectron // 2
            mo_energy_alpha = mf.mo_energy
            homo = mo_energy_alpha[nocc-1]
            lumo = mo_energy_alpha[nocc]
        else:
            mf = dft.UKS(mol)
            mf.xc = xc_functional
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
    
    def get_j_function(self, alpha=None, beta=None, omega=None):

        E_N, E_homo, E_lumo = self.calculate_energy(charge=0, spin=0, alpha=alpha, beta=beta, omega=omega)
        E_N_plus_1, E_homo_plus_1, E_lumo_plus_1 = self.calculate_energy(charge=-1, spin=1, alpha=alpha, beta=beta, omega=omega)
        E_N_minus_1, E_homo_minus_1, E_lumo_minus_1 = self.calculate_energy(charge=1, spin=1, alpha=alpha, beta=beta, omega=omega)
        
        J_N = abs(E_homo + E_N_minus_1 - E_N)
        J_N_plus_1 = abs(E_homo_plus_1 + E_N - E_N_plus_1)
        J = J_N**2 + J_N_plus_1**2
        return J
    
    def forward(self, alpha=None, beta=None, omega=None):
        """允许在forward时传入不同的参数"""
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if omega is None:
            omega = self.omega
            
        J = self.get_j_function(alpha=alpha, beta=beta, omega=omega)
        return J, alpha, beta, omega

if __name__ == '__main__':
    mol2 = '/Users/jiaoyuan/Documents/GitHub/ADOPTXC/module/net.mol2'
    energy_getter = EnergyGetter(mol2)
    J = energy_getter.get_j_function()
    energy_getter.forward()
    print(J)