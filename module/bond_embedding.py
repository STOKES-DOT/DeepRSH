from __future__ import annotations
import os; os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
from dataclasses import dataclass
from typing import List, Optional, Tuple
import jax
import jax.numpy as jnp



@jax.tree_util.register_pytree_node_class
@dataclass
class GaussianBasisParams:
    log_sigma: jnp.ndarray
    mu: jnp.ndarray

    @property
    def sigma(self) -> jnp.ndarray:
        return jnp.exp(self.log_sigma)

    def tree_flatten(self):
        children = (self.log_sigma, self.mu)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        log_sigma, mu = children
        return cls(log_sigma=log_sigma, mu=mu)


@dataclass
class BondEmbedding:
    mol2: str
    files: Optional[str] = None
    index: Optional[int] = None

    def _read_atoms(self) -> List[Tuple[int, str, float, float, float]]:
        atoms: List[Tuple[int, str, float, float, float]] = []
        with open(self.mol2, "r") as f:
            lines = f.readlines()

        atom_section = False
        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                atom_section = True
                continue
            elif line.startswith("@<TRIPOS>"):
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
                    atoms.append((atom_index, atom_name, x, y, z))
        return atoms

    def _read_bonds(self) -> Tuple[List[int], List[int], List[float]]:
        bond_type: List[float] = []
        atom1: List[int] = []
        atom2: List[int] = []

        with open(self.mol2, "r") as f:
            lines = f.readlines()

        bond_section = False
        for line in lines:
            if line.startswith("@<TRIPOS>BOND"):
                bond_section = True
                continue
            elif line.startswith("@<TRIPOS>"):
                bond_section = False
                continue

            if bond_section and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    bt = str(parts[3])
                    a1 = int(parts[1])
                    a2 = int(parts[2])
                    atom1.append(a1)
                    atom2.append(a2)
                    bond_type.append(self._map_bond_type(bt))

        return atom1, atom2, bond_type

    @staticmethod
    def _map_bond_type(bt: str) -> float:
        if bt == "1":
            return 1.0
        if bt == "2":
            return 2.0
        if bt == "3":
            return 3.0
        if bt == "ar":
            return 1.5
        if bt == "am":
            return 1.2
        raise ValueError(f"Unsupported bond type: {bt}")

    @staticmethod
    def gaussian_basis_matrix_jax(
        coords: jnp.ndarray,
        gb_params: GaussianBasisParams,
        cutoff: float = 20.0,
        eps: float = 1e-12,
    ) -> jnp.ndarray:
        diff = coords[:, None, :] - coords[None, :, :]
        rij2 = jnp.sum(diff * diff, axis=-1)
        r = jnp.sqrt(rij2 + jnp.asarray(eps, dtype=rij2.dtype) )

        sigma = gb_params.sigma
        mu = gb_params.mu

        norm = jnp.array(1.0, dtype=sigma.dtype) / (jnp.sqrt(jnp.array(2.0, dtype=sigma.dtype) * jnp.pi) * sigma)
        val = norm * jnp.exp(-((r - mu) ** 2) / (jnp.array(2.0, dtype=sigma.dtype) * sigma**2))
        cutoff_jnp = jnp.asarray(cutoff, dtype=r.dtype)  
        gb = jnp.where(r <= cutoff_jnp, val, 0.0)
        return gb.astype(jnp.float32)

    def atom_coords(self) -> jnp.ndarray:
        atoms = self._read_atoms()
        coords = jnp.array([(x, y, z) for (_, _, x, y, z) in atoms], dtype=jnp.float32)
        return coords

    def get_bond_type(self) -> jnp.ndarray:
        atom1, atom2, bond_type = self._read_bonds()
        if len(atom1) == 0:
            raise ValueError("No bonds found in mol2 @<TRIPOS>BOND section.")
        max_index = max(max(atom1), max(atom2))
        bond_type_matrix = jnp.zeros((max_index, max_index), dtype=jnp.float32)
        for bt, (i, j) in zip(bond_type, zip(atom1, atom2)):
            bond_type_matrix = bond_type_matrix.at[i - 1, j - 1].set(bt)
            bond_type_matrix = bond_type_matrix.at[j - 1, i - 1].set(bt)
        return bond_type_matrix

    def get_degree_matrix(self) -> jnp.ndarray:
        bond_type_matrix = self.get_bond_type()
        return (bond_type_matrix != 0).astype(jnp.float32)

    def get_atom_pairs_direction(self) -> List[Tuple[float, float, float]]:
        atoms = self._read_atoms()
        directions: List[Tuple[float, float, float]] = []
        n_atoms = len(atoms)
        for i in range(n_atoms):
            for j in range(n_atoms):
                _, _, x1, y1, z1 = atoms[i]
                _, _, x2, y2, z2 = atoms[j]
                directions.append((x2 - x1, y2 - y1, z2 - z1))
        return directions

    def __call__(self, gb_params: GaussianBasisParams, cutoff: float = 20.0):
        coords = jnp.asarray(self.atom_coords(), dtype=jnp.float32)
        gb_matrix = self.gaussian_basis_matrix_jax(coords, gb_params, cutoff=cutoff)
        bond_type_matrix = jnp.asarray(self.get_bond_type(), dtype=jnp.float32)
        degree_matrix = jnp.asarray(self.get_degree_matrix(), dtype=jnp.float32)
        direction = jnp.asarray(np.array(self.get_atom_pairs_direction(), dtype=np.float32), dtype=jnp.float32)
        return gb_matrix, bond_type_matrix, degree_matrix, direction


def init_gaussian_basis_params(init_sigma: float = 1.0, init_mu: float = 0.0) -> GaussianBasisParams:
    return GaussianBasisParams(
        log_sigma=jnp.log(jnp.asarray(init_sigma, dtype=jnp.float32)),
        mu=jnp.asarray(init_mu, dtype=jnp.float32),
    )


if __name__ == "__main__":
    mol2 = "/Users/jiaoyuan/Documents/GitHub/DeepRSH/DeepRSH/module/net.mol2"
    be = BondEmbedding(mol2)
    print("Reading molecule from:", mol2)
    print("Number of atoms:", be)
    gb_params = init_gaussian_basis_params(init_sigma=1.0, init_mu=0.0)

    def loss_fn(p: GaussianBasisParams):
        gb, _, _, _ = be(p)
        return jnp.sum(gb)

    grads = jax.grad(loss_fn)(gb_params)
    print("dL/dlog_sigma:", grads.log_sigma)
    print("dL/dmu:", grads.mu)
    
    
    
    
