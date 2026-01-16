import os
import pandas as pd
from tqdm import tqdm
from pyscf import gto
from typing import List, Tuple, Optional
from grad_dft import saver as save, molecule_from_pyscf
from grad_dft.interface.pyscf import process_mol
from pyscf import gto, scf, lib
import jax
import jax.numpy as jnp
lib.num_threads(96)
lib.logger.QUIET = True
# 默认配置
DEFAULT_BASIS = "cc-pvtz"
DEFAULT_GRID_LEVEL = 2

def process_h2_energy_from_csv(
    csv_path: str,
    basis: str = DEFAULT_BASIS,
    grid_level: int = DEFAULT_GRID_LEVEL,
    spin: int = 0,
    training: bool = True,
    max_cycle: Optional[int] = 50,
    save_path: Optional[str] = None,
) -> Tuple[List, List, List]:
    """
    专门处理 H2 分子 CSV 文件的函数。
    CSV 格式假设：
    - 第一列：H-H 键长 (Angstrom 或 Bohr，通常 PySCF 默认为 Angstrom)
    - 第二列：FCI 或其他参考能量 (Hartree)
    
    Args:
        csv_path: CSV 文件路径
        ... (其他参数保持不变)
    """
    try:
        df = pd.read_csv(csv_path, header=None) 
    except Exception as e:
        raise RuntimeError(f"Could not read CSV file {csv_path}: {e}")

    molecules_neutral = []
    molecules_cation = [] 
    molecules_anion = []  
    omegas = [0.0, 0.3]
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing H2 Curve"):
        
        try:

            bond_length = float(row[0])
            target_energy_neutral = float(row[1])
            atom_str = f"H 0 0 0; H 0 0 {bond_length}"

            name = f"h2_{bond_length:.3f}"

            mol = gto.M(atom=atom_str, basis=basis, charge=0, spin=spin)
            mol.max_memory = 20000
            _, mf_n = process_mol(
                mol, compute_energy=False, grid_level=grid_level, 
                training=training, max_cycle=0
            )
            
            molecules_neutral.append(molecule_from_pyscf(
                mf_n, name=f"{name}_neutral", energy=target_energy_neutral, scf_iteration=max_cycle, omegas=omegas
            ))

            # cation_spin = spin + 1
            # cation_mol = gto.M(atom=atom_str, basis=basis, charge=1, spin=cation_spin)
            # cation_mol.max_memory = 20000
            # energy_c, mf_c = process_mol(
            #     cation_mol, compute_energy=False, grid_level=grid_level, 
            #     training=training, max_cycle=0
            # )
            
            # molecules_cation.append(molecule_from_pyscf(
            #     mf_c, name=f"{name}_cation", energy=energy_c, omegas=omegas
            # ))

            # anion_spin = spin + 1
            # anion_mol = gto.M(atom=atom_str, basis=basis, charge=-1, spin=anion_spin)
            # anion_mol.max_memory = 20000
            # energy_a, mf_a = process_mol(
            #     anion_mol, compute_energy=False, grid_level=grid_level, 
            #     training=training, max_cycle=0
            # )
            
            # molecules_anion.append(molecule_from_pyscf(
            #     mf_a, name=f"{name}_anion", energy=energy_a, omegas=omegas
            # ))

        except Exception as e:
            print(f"Failed to process row {index} (R={row[0] if len(row)>0 else 'N/A'}): {e}")
            continue

    if save_path:
        print(f"Saving processed data to {save_path}...")
        all_mols = molecules_neutral #+ molecules_cation + molecules_anion
        save(molecules=all_mols, fname=save_path)

    return molecules_neutral#, molecules_cation, molecules_anion


csv_file_path = "/home/yjiao/DeepRSH/dataset_diatoms/h2_fci_curve.csv"
output_h5_path = "h2_dataset_n.h5"
neutral = process_h2_energy_from_csv(
        csv_path=csv_file_path,
        save_path=output_h5_path,    
        basis="cc-pvtz",              
        grid_level=2,               
        training=True              
    )
all_molecules = neutral #+ cation + anion
manual_save_path = "h2_dataset_n_manual.h5"
save(molecules=all_molecules, fname=manual_save_path)