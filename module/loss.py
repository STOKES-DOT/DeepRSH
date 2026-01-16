from typing import Callable, Tuple, Dict, Union
from functools import partial
from jaxtyping import PyTree, Array, Scalar

import jax.numpy as jnp
from jax import value_and_grad, scipy as jsp
from jax.tree_util import tree_map

from grad_dft import Molecule

def get_homo_energy(fock: Array, molecule: Molecule) -> Scalar:
    """
    Calculates the HOMO energy from the Fock matrix.
    
    Solves the generalized eigenvalue problem F C = E S C to find orbital energies,
    then selects the energy of the highest occupied molecular orbital.
    """
    # Solve generalized eigenvalue problem: F @ C = E @ S @ C
    # molecular orbitals energies (w) are the eigenvalues
    w, _ = jsp.linalg.eigh(fock, b=molecule.s1e)
    
    # Identify HOMO index
    # Assumes mo_occ is sorted (occupied -> virtual) and contains 
    # approx 2.0 for occupied and 0.0 for virtual orbitals.
    # We find the index of the last orbital with non-zero occupation.
    n_occ = jnp.sum(molecule.mo_occ > 1e-1).astype(int)
    homo_index = n_occ - 1
    
    return w[homo_index]

def janak_loss(
    params: PyTree,
    compute_energy: Callable,
    molecule: Molecule,
    cation_molecule: Molecule,
    anion_molecule: Molecule
) -> Scalar:
    """
    Computes the Janak loss based on the piecewise linearity condition of the energy.
    
    Enforces the condition that the HOMO energy should equal the negative 
    Ionization Potential (IP) for both the neutral and anion systems.
    
    Args:
        params: Functional parameters.
        compute_energy: Function returning (energy, fock_matrix).
        molecule: Neutral molecule (N electrons).
        cation_molecule: Cation molecule (N-1 electrons).
        anion_molecule: Anion molecule (N+1 electrons).
    """
    
    # Compute energies and Fock matrices
    E_N, fock_N = compute_energy(params, molecule)
    E_Np1, fock_Np1 = compute_energy(params, anion_molecule)
    E_Nm1, _ = compute_energy(params, cation_molecule)

    # Calculate HOMO energies
    homo_N = get_homo_energy(fock_N, molecule)
    homo_Np1 = get_homo_energy(fock_Np1, anion_molecule)

    # Janak's Theorem / Koopmans' compliant conditions:
    # 1. For N system: HOMO(N) = E(N) - E(N-1) => HOMO(N) + E(N-1) - E(N) = 0
    J_N = jnp.abs(homo_N + E_Nm1 - E_N)
    
    # 2. For N+1 system: HOMO(N+1) = E(N+1) - E(N) => HOMO(N+1) + E(N) - E(N+1) = 0
    J_Np1 = jnp.abs(homo_Np1 + E_N - E_Np1)

    return J_N**2 + J_Np1**2

@partial(value_and_grad, has_aux=True)
def total_loss(
    params: PyTree,
    compute_energy: Callable,
    molecule: Molecule,
    cation_molecule: Molecule, # Corresponds to 'anion_mol' (N-1) in original code
    anion_molecule: Molecule,  # Corresponds to 'ion_mol' (N+1) in original code
    ground_truth_energy: Scalar,
    janak_weight: float = 1.0
) -> Tuple[Scalar, Dict[str, Scalar]]:
    """
    Computes the total loss combining MSE energy loss and Janak loss.
    """
    
    # Compute Janak loss component
    janak_loss_val = janak_loss(
        params, 
        compute_energy, 
        molecule, 
        cation_molecule, 
        anion_molecule
    )
    
    # Compute Energy prediction for neutral molecule
    predicted_energy, _ = compute_energy(params, molecule)
    
    # MSE Energy Loss
    mse_cost = (predicted_energy - ground_truth_energy) ** 2

    # Total Cost
    total_cost = mse_cost + janak_weight * janak_loss_val

    metrics = {
        "predicted_energy": predicted_energy,
        "ground_truth_energy": ground_truth_energy,
        "mean_abs_error": jnp.abs(predicted_energy - ground_truth_energy),
        "mean_sq_error": mse_cost,
        "janak_loss": janak_loss_val,
        "total_loss": total_cost,
    }

    return total_cost, metrics