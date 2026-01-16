from __future__ import annotations
import os; os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AtomEmbedding:
    mol2: str
    atom_type_dict: str = "/home/yjiao/DeepRSH/module/atom_embedding_shell.dat"

    def get_atom_type(self) -> List[str]:
        atom_types: List[str] = []
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
                    atom_symbol = parts[1][:2].title()
                    atom_types.append(atom_symbol)
        return atom_types

    def atom_type_to_vector(self) -> List[List[int]]:
        atom_types = self.get_atom_type()

        with open(self.atom_type_dict, "r") as f:
            lines = f.readlines()

        atom_vector_map = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(", [", 1)
            if len(parts) < 2:
                continue

            atom_symbol = parts[0].strip()
            vector_str = parts[1].rstrip("]")
            vector = [int(x) for x in vector_str.split(",")]
            atom_vector_map[atom_symbol] = vector

        result_vectors: List[List[int]] = []
        for atom in atom_types:
            if atom in atom_vector_map:
                result_vectors.append(atom_vector_map[atom])
            else:
                raise ValueError(f"Atom type {atom} not found in the dictionary")
        return result_vectors

    def atom_molecular_part(self) -> List[int]:
        atom_molecule_part: List[int] = []
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
                    atom_symbol = parts[6][:2].title()
                    atom_molecule_part.append(int(atom_symbol))
        return atom_molecule_part

    def atom_charge(self) -> List[float]:
        atom_charge: List[float] = []
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
                    atom_symbol = parts[8][:-1]
                    atom_charge.append(float(atom_symbol))
        return atom_charge

    def __call__(self) -> Tuple[List[List[int]], List[int], List[float]]:
        atom_type_vector = self.atom_type_to_vector()
        atom_part = self.atom_molecular_part()
        atom_charge = self.atom_charge()
        return atom_type_vector, atom_part, atom_charge


if __name__ == "__main__":
    mol2 = "/Users/jiaoyuan/Documents/GitHub/DeepRSH/DeepRSH/module/net.mol2"
    atom_embed = AtomEmbedding(mol2)
    atom_type_vector, atom_part, atom_charge = atom_embed()
    print(atom_type_vector)
    print(atom_part)
    print(atom_charge)