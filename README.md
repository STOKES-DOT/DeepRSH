# DeepRSH: Deep Learning Range-Separated Hybrid Functionals

A deep learning framework for training range-separated hybrid (RSH) density functionals using JAX and grad_dft.

## Overview

DeepRSH implements neural network-based exchange-correlation functionals that combine:
- Traditional XC energy densities (LDA, B88, VWN, LYP)
- Hartree-Fock exact exchange with range separation
- Learned mixing coefficients via neural networks

## Features

- **Range-Separated Hybrids**: Supports multiple omega parameters for long-range exact exchange
- **Trainable Functionals**: End-to-end differentiable implementation using JAX
- **GAT Architecture**: Graph Attention Networks for molecular representation
- **H₂ Dissociation**: Specialized datasets for diatomic molecules

## Installation

```bash
# Install dependencies
pip install jax jaxlib flax optax grad_dft pyscf

# Or use conda
conda create -n deeprsh python=3.10
conda activate deeprsh
pip install -r requirements.txt
```

## Usage

### Creating a Functional

```python
from xc_functional import rsh_b3lyp_nn
from grad_dft import energy_predictor

# Create the functional
functional = rsh_b3lyp_nn()

# Create energy predictor
compute_energy = energy_predictor(functional)

# Predict energy and Fock matrix
energy, fock = compute_energy(params, molecule)
```

### Training

```python
from grad_dft import train_kernel, loader
from optax import adam
from functools import partial
from jax import value_and_grad
import jax.numpy as jnp

# Load data with omegas
omegas = jnp.array([0.0, 0.3])
dataset = loader(fname="data.h5", config_omegas=omegas)

# Define loss
@partial(value_and_grad, has_aux=True)
def loss(params, molecule, true_energy):
    predicted_energy, fock = compute_energy(params, molecule)
    cost_value = (predicted_energy - true_energy) ** 2
    return cost_value, {"energy": predicted_energy}

# Train
tx = adam(learning_rate=1e-4)
kernel = jax.jit(train_kernel(tx, loss))
```

## Project Structure

```
DeepRSH/
├── xc_functional.py       # RSH functional definition
├── train.py              # Training script
├── DeepsRSHXC.py         # GAT-based model
├── data_process.py       # Data preprocessing
├── GAT_Layer.py          # Graph Attention layers
├── nodes_embedding.py    # Node embeddings
├── bond_embedding.py     # Bond embeddings
└── dataset_diatoms/      # Diatomic molecule datasets
```

## Range-Separated Parameters

The default omegas are `[0.0, 0.3]`:
- `ω = 0.0`: Pure DFT (no exact exchange)
- `ω = 0.3`: 30% long-range exact exchange (LC-ωPBE style)

## Citation

If you use this code, please cite:

```bibtex
@software{deeprsh2024,
  title = {DeepRSH: Deep Learning Range-Separated Hybrid Functionals},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/STOKES-DOT/DeepRSH}
}
```

## License

MIT License

