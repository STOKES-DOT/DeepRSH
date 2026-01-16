#!/usr/bin/env python
"""
H2 dissociation curve: FCI / def2-QZVP
1) 保存能量到 csv
2) 每个构型输出 .xyz 文件（纯 PySCF，无外部依赖）
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pyscf import gto, scf, fci, ao2mo
from pyscf import lib

lib.num_threads(96)
# --------------- 用户参数 ---------------
basis_name = 'cc-pvtz'          # 最大 def2 基组
rmin, rmax, dr = 0.1, 4, 0.1
csv_out  = 'h2_fci_curve.csv'
xyz_dir  = 'xyz_files'            # 专门放 xyz 的文件夹
# ---------------------------------------

os.makedirs(xyz_dir, exist_ok=True)

def write_xyz(mol, xyz_path):
    """把 pyscf Mole 对象写成 xyz"""
    with open(xyz_path, 'w') as f:
        f.write(f"{mol.natm}\n")
        f.write(f"H2  r = {mol.atom_coord(1)[2]:.2f} A   FCI/{basis_name}\n")
        for i in range(mol.natm):
            symb = mol.atom_symbol(i)
            x, y, z = mol.atom_coord(i)
            f.write(f"{symb:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")

distances, energies = [], []

for d in np.arange(rmin, rmax + dr, dr):
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {d:.2f}',
        basis=basis_name,
        symmetry=True,
        verbose=0
    )
    mol.max_memory = 20000
    mf = scf.RHF(mol).run()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mol, mf.mo_coeff)
    e_fci = fci.FCI(mf).kernel(h1, h2, mol.nao, mol.nelectron)[0]

    distances.append(d)
    energies.append(e_fci)

    # ---------- 输出 xyz ----------
    xyz_path = os.path.join(xyz_dir, f'h2_{d*100:03.0f}pm.xyz')  # 如 120pm.xyz
    write_xyz(mol, xyz_path)

    print(f'r = {d:4.2f} Å   E(FCI) = {e_fci:18.10f} Ha  -> {xyz_path}')

# 保存 csv
df = pd.DataFrame({'r_Ang': distances, 'E_FCI_Ha': energies})
df.to_csv(csv_out, index=False, float_format='%.8f')
print(f'\n能量曲线已写入 {csv_out}')
print(f'全部 xyz 文件保存在 ./{xyz_dir}/')

# 简单作图
plt.plot(distances, energies, 'o-', label=f'FCI/{basis_name}')
plt.xlabel('Bond length / Å')
plt.ylabel('Energy / Ha')
plt.legend()
plt.title('H$_2$ dissociation curve')
plt.tight_layout()
plt.savefig('h2_fci_curve.png', dpi=150)
plt.show()
