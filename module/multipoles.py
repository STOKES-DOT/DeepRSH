import numpy as np
from pyscf import gto, scf, tools
from pyscf.data.elements import charge

def distributed_multipole_analysis(mol, dm, grid_level=3, lmax=4):

    # 生成Lebedev网格点用于数值积分
    from pyscf.dft import gen_grid
    grid = gen_grid.Grids(mol)
    grid.level = grid_level
    grid.build()
    
    # 计算分子轨道在网格点上的值
    from pyscf.dft import numint
    ao_value = numint.eval_ao(mol, grid.coords, deriv=0)
    
    # 计算电子密度在网格点上的值
    rho = numint.eval_rho(mol, ao_value, dm)
    
    # 获取原子坐标
    coords = mol.atom_coords()
    charges = [charge(a[0]) for a in mol._atom]
    
    multipoles = {}
    
    for iatom in range(mol.natm):
        atom_coord = coords[iatom]
        atom_charge = charges[iatom]
        
        # 计算相对于当前原子的坐标
        r_vec = grid.coords - atom_coord
        r = np.linalg.norm(r_vec, axis=1)
        
        # 初始化该原子的多极矩
        atom_multipoles = {}
        
        # 计算各阶多极矩
        for l in range(lmax + 1):
            if l == 0:  # 单极矩（电荷）
                # 电子部分
                Y00 = 1.0 / np.sqrt(4 * np.pi)
                monopole_elec = -np.sum(rho * Y00 * grid.weights)
                # 核部分
                monopole_nuc = atom_charge
                atom_multipoles['monopole'] = monopole_elec + monopole_nuc
                
            elif l == 1:  # 偶极矩
                dipole = np.zeros(3)
                for i in range(3):
                    # 电子部分
                    dipole_elec = -np.sum(rho * r_vec[:, i] * grid.weights)
                    # 核部分 (对于偶极矩，核部分为0，因为以原子为中心)
                    dipole[i] = dipole_elec
                atom_multipoles['dipole'] = dipole
                
            elif l == 2:  # 四极矩
                quadrupole = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        # 电子部分
                        quad_elec = -np.sum(rho * (3 * r_vec[:, i] * r_vec[:, j] - 
                                                  (i == j) * r**2) * grid.weights)
                        quadrupole[i, j] = quad_elec
                atom_multipoles['quadrupole'] = quadrupole
                
            elif l == 3:  # 八极矩
                octupole = np.zeros((3, 3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            term = (5 * r_vec[:, i] * r_vec[:, j] * r_vec[:, k] -
                                   (r**2) * (r_vec[:, i] * (j == k) + 
                                            r_vec[:, j] * (i == k) + 
                                            r_vec[:, k] * (i == j)))
                            oct_elec = -np.sum(rho * term * grid.weights)
                            octupole[i, j, k] = oct_elec
                atom_multipoles['octupole'] = octupole
                
            elif l == 4:  # 十六极矩
                hexadecapole = np.zeros((3, 3, 3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for m in range(3):
                                # 十六极矩的表达式
                                term = (35 * r_vec[:, i] * r_vec[:, j] * r_vec[:, k] * r_vec[:, m] -
                                       5 * r**2 * (r_vec[:, i] * r_vec[:, j] * (k == m) +
                                                  r_vec[:, i] * r_vec[:, k] * (j == m) +
                                                  r_vec[:, i] * r_vec[:, m] * (j == k) +
                                                  r_vec[:, j] * r_vec[:, k] * (i == m) +
                                                  r_vec[:, j] * r_vec[:, m] * (i == k) +
                                                  r_vec[:, k] * r_vec[:, m] * (i == j)) +
                                       r**4 * ((i == j) * (k == m) + 
                                              (i == k) * (j == m) + 
                                              (i == m) * (j == k)))
                                hex_elec = -np.sum(rho * term * grid.weights)
                                hexadecapole[i, j, k, m] = hex_elec
                atom_multipoles['hexadecapole'] = hexadecapole
        
        multipoles[iatom] = atom_multipoles
    
    return multipoles

def print_multipole_analysis(mol, multipoles):
    """打印多极矩分析结果"""
    elements = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords = mol.atom_coords()
    
    print("=" * 80)
    print("分布式多极矩分析 (DMA) 结果")
    print("=" * 80)
    
    for iatom in range(mol.natm):
        print(f"\n原子 {iatom+1}: {elements[iatom]} 在坐标 {coords[iatom]}")
        print("-" * 50)
        
        mp = multipoles[iatom]
        
        # 单极矩（净电荷）
        print(f"单极矩 (电荷): {mp['monopole']:.6f} e")
        
        # 偶极矩
        dipole = mp['dipole']
        dipole_norm = np.linalg.norm(dipole)
        print(f"偶极矩: [{dipole[0]:.6f}, {dipole[1]:.6f}, {dipole[2]:.6f}] e·Å")
        print(f"偶极矩模长: {dipole_norm:.6f} e·Å")
        
        # 四极矩
        quad = mp['quadrupole']
        print("四极矩 (e·Å²):")
        for i in range(3):
            print(f"  [{quad[i,0]:.6f}, {quad[i,1]:.6f}, {quad[i,2]:.6f}]")
        
        # 八极矩
        oct = mp['octupole']
        print("八极矩 (e·Å³):")
        for i in range(3):
            for j in range(3):
                print(f"  [{oct[i,j,0]:.6f}, {oct[i,j,1]:.6f}, {oct[i,j,2]:.6f}]")
            if i < 2:
                print("   ---")
        
        # 十六极矩
        hexa = mp['hexadecapole']
        print("十六极矩 (e·Å⁴):")
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    print(f"    [{hexa[i,j,k,0]:.6f}, {hexa[i,j,k,1]:.6f}, {hexa[i,j,k,2]:.6f}]")
                if k < 2:
                    print("     ---")
            if j < 2:
                print("   ---")
        if i < 2:
            print("  ---")

# 示例使用
def main():
    # 创建水分子
    mol = gto.Mole()
    mol.atom = '''
    O 0.0 0.0 0.0
    H 0.757 0.586 0.0
    H -0.757 0.586 0.0
    '''
    mol.basis = 'sto-3g'
    mol.build()
    
    # 进行HF计算
    mf = scf.RHF(mol)
    mf.kernel()
    
    # 获取密度矩阵
    dm = mf.make_rdm1()
    
    # 进行分布式多极矩分析
    print("正在进行分布式多极矩分析...")
    multipoles = distributed_multipole_analysis(mol, dm, grid_level=4, lmax=4)
    
    # 打印结果
    print_multipole_analysis(mol, multipoles)
    
    # 可选：保存结果到文件



if __name__ == "__main__":
    main()