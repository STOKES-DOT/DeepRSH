# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from jax.random import split, PRNGKey
from jax import numpy as jnp, value_and_grad
from jax.nn import gelu
import numpy as np
from optax import adam
from tqdm import tqdm
import os
import pickle
import json
import matplotlib.pyplot as plt
from orbax.checkpoint import PyTreeCheckpointer
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
from grad_dft import (
    train_kernel, 
    energy_predictor,
    NeuralFunctional,
    canonicalize_inputs,
    dm21_coefficient_inputs,
    dm21_densities,
    loader
)
from grad_dft.data_processing import process_dissociation
from torch.utils.tensorboard import SummaryWriter
import jax
from xc_functional import rsh_b3lyp_nn

# In this example we explain how to replicate the experiments that train
# the functional in some points of the dissociation curve of H2 or H2^+.
distances = [0.5, 0.75, 1, 1.25, 1.5]
# process_dissociation(atom1 = 'H', atom2 = 'H', spin = 0, file = 'H2_dissociation.xlsx', energy_column_name='sto-3g', training_distances=distances)
dirpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
training_data_dirpath = os.path.normpath(dirpath + "/data/training/dissociation/")
training_files = ["H2_extrapolation_train.h5"]

####### Model definition #######
functional = rsh_b3lyp_nn()
print(functional)

####### Initializing the functional and some parameters #######
key = PRNGKey(42)  # Jax-style random seed

# We generate the features from the molecule we created before, to initialize the parameters
(key,) = split(key, 1)
rhoinputs = jax.random.normal(key, shape=[2, 7])
params = functional.init(key, rhoinputs)

checkpoint_step = 0
learning_rate = 1e-4
momentum = 0.9
tx = adam(learning_rate=learning_rate, b1=momentum)
opt_state = tx.init(params)
cost_val = jnp.inf

ckpt_dir = os.path.join(dirpath, "ckpts/H2_extrapolation", "checkpoint_" + str(checkpoint_step) + "/")
loadcheckpoint = False
if loadcheckpoint:
    # Note: Need to fix the orbax_checkpointer reference here
    orbax_checkpointer = PyTreeCheckpointer()
    train_state = functional.load_checkpoint(
        tx=tx, ckpt_dir=ckpt_dir, step=checkpoint_step, orbax_checkpointer=orbax_checkpointer
    )
    params = train_state.params
    tx = train_state.tx
    opt_state = tx.init(params)
    epoch = train_state.step

########### Definition of the loss function #####################

# Here we use one of the following. We will use the second here.
compute_energy = energy_predictor(functional)

def save_checkpoint_safe(params, opt_state, step, base_dir):
    """Safely save checkpoint, handling JAX arrays"""
    import pickle
    import json
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(base_dir, f"epoch_{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Convert JAX arrays to numpy arrays
    def to_numpy(x):
        if hasattr(x, '__array__'):
            return np.array(x)
        return x
    
    params_np = jax.tree_map(to_numpy, params)
    opt_state_np = jax.tree_map(to_numpy, opt_state)
    
    # Save parameters
    with open(os.path.join(checkpoint_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params_np, f)
    
    # Save optimizer state
    with open(os.path.join(checkpoint_dir, 'opt_state.pkl'), 'wb') as f:
        pickle.dump(opt_state_np, f)
    
    # Save metadata
    metadata = {
        'step': step,
        'learning_rate': lr,
        'momentum': momentum
    }
    with open(os.path.join(checkpoint_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    print(f"Checkpoint saved at epoch {step} in {checkpoint_dir}")
    return checkpoint_dir

@partial(value_and_grad, has_aux=True)
def loss(params, molecule, true_energy):
    # In general the loss function should be able to accept [params, system (eg, molecule or reaction), true_energy]
    predicted_energy, fock = compute_energy(params, molecule)
    cost_value = (predicted_energy - true_energy) ** 2

    # We may want to add a regularization term to the cost, be it one of the
    # fock_grad_regularization, dm21_grad_regularization, or orbital_grad_regularization in train.py;
    # or even the satisfaction of the constraints in constraints.py.

    metrics = {
        "predicted_energy": predicted_energy,
        "ground_truth_energy": true_energy,
        "mean_abs_error": jnp.mean(jnp.abs(predicted_energy - true_energy)),
        "mean_sq_error": jnp.mean((predicted_energy - true_energy) ** 2),
        "cost_value": cost_value,
    }

    cost_value = cost_value

    return cost_value, metrics

kernel = jax.jit(train_kernel(tx, loss))

# Create directory to save training metrics
metrics_dir = os.path.join(dirpath, "training_metrics")
os.makedirs(metrics_dir, exist_ok=True)

######## Training epoch ########
def train_epoch(state, training_files, training_data_dirpath):
    """Train for a single epoch."""
    batch_metrics = []
    params, opt_state, cost_val = state
    
    for file in tqdm(training_files, "Files"):
        fpath = os.path.join(training_data_dirpath, file)
        print("Training on file: ", fpath, "\n")

        load = loader(fname=fpath, randomize=True, training=True, config_omegas=[])
        for _, system in tqdm(load, "Molecules/reactions per file"):
            params, opt_state, cost_val, metrics = kernel(params, opt_state, system, system.energy)
            del system
            batch_metrics.append(metrics)

    epoch_metrics = {
        k: np.mean([jax.device_get(metrics[k]) for metrics in batch_metrics])
        for k in batch_metrics[0]
    }
    state = (params, opt_state, cost_val)
    return state, batch_metrics, epoch_metrics

######## Evaluation function ########
def evaluate_model(params, data_files, data_dir):
    """Evaluate model performance on dataset and return per-molecule details with bond lengths"""
    compute_energy = energy_predictor(functional)
    
    all_results = []  # Store results for each molecule
    
    for file in data_files:
        fpath = os.path.join(data_dir, file)
        print(f"Evaluating on file: {fpath}")
        
        load = loader(fname=fpath, randomize=False, training=False, config_omegas=[])
        for _, system in tqdm(load, "Evaluating molecules"):
            predicted_energy, _ = compute_energy(params, system)
            
            # Extract bond length for H2 molecule (assuming two hydrogen atoms)
            # For H2, bond length is the distance between the two atoms
            if hasattr(system, 'coords') and len(system.coords) == 2:
                # Assuming system.coords is in atomic units (Bohr)
                coords = jax.device_get(system.coords)
                bond_length = np.linalg.norm(coords[0] - coords[1])
            else:
                bond_length = None
            
            result = {
                "bond_length": bond_length,
                "true_energy": jax.device_get(system.energy) if hasattr(system.energy, '__array__') else system.energy,
                "predicted_energy": jax.device_get(predicted_energy),
                "absolute_error": abs(jax.device_get(predicted_energy) - (jax.device_get(system.energy) if hasattr(system.energy, '__array__') else system.energy)),
            }
            all_results.append(result)
    
    # Calculate overall statistics
    if all_results:
        mae = np.mean([r["absolute_error"] for r in all_results])
        mse = np.mean([r["absolute_error"]**2 for r in all_results])
        rmse = np.sqrt(mse)
        
        overall_metrics = {
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "max_absolute_error": np.max([r["absolute_error"] for r in all_results]),
            "min_absolute_error": np.min([r["absolute_error"] for r in all_results]),
            "num_samples": len(all_results)
        }
        
        return overall_metrics, all_results
    return None, []

######## Visualization functions ########
def plot_training_curves(history, save_path):
    """Plot training curves with English labels"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    epochs = list(range(len(history)))
    losses = [h.get('cost_value', 0) for h in history]
    mae = [h.get('mean_abs_error', 0) for h in history]
    mse = [h.get('mean_sq_error', 0) for h in history]
    
    # 1. Loss function curve
    axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training Loss Curve')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')  # Use log scale for better visualization
    
    # 2. Mean absolute error curve
    axes[0, 1].plot(epochs, mae, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error (Ha)')
    axes[0, 1].set_title('Mean Absolute Error Curve')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Mean squared error curve
    axes[1, 0].plot(epochs, mse, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean Squared Error (Ha²)')
    axes[1, 0].set_title('Mean Squared Error Curve')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 4. Loss vs MAE correlation
    axes[1, 1].scatter(losses, mae, alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Loss (MSE)')
    axes[1, 1].set_ylabel('Mean Absolute Error (Ha)')
    axes[1, 1].set_title('Loss vs MAE Correlation')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save loss curve separately
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve for H₂ Dissociation')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_dissociation_curve(all_results, save_path):
    """Plot dissociation curve for H₂ molecule"""
    # Filter results with bond lengths
    valid_results = [r for r in all_results if r["bond_length"] is not None]
    
    if not valid_results:
        print("No bond length information available for dissociation curve")
        return
    
    # Sort by bond length
    valid_results.sort(key=lambda x: x["bond_length"])
    
    bond_lengths = [r["bond_length"] for r in valid_results]
    true_energies = [r["true_energy"] for r in valid_results]
    predicted_energies = [r["predicted_energy"] for r in valid_results]
    errors = [r["absolute_error"] for r in valid_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Dissociation curve: True and predicted energies
    axes[0, 0].plot(bond_lengths, true_energies, 'b-', marker='o', linewidth=2, markersize=6, label='True Energy (Reference)')
    axes[0, 0].plot(bond_lengths, predicted_energies, 'r--', marker='s', linewidth=2, markersize=6, label='Predicted Energy (NN-XC)')
    axes[0, 0].set_xlabel('Bond Length (Bohr)')
    axes[0, 0].set_ylabel('Total Energy (Ha)')
    axes[0, 0].set_title('H₂ Dissociation Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Energy error vs bond length
    axes[0, 1].plot(bond_lengths, errors, 'g-', marker='^', linewidth=2, markersize=6)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Bond Length (Bohr)')
    axes[0, 1].set_ylabel('Absolute Error (Ha)')
    axes[0, 1].set_title('Prediction Error vs Bond Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Predicted vs True energy scatter plot
    axes[1, 0].scatter(true_energies, predicted_energies, alpha=0.7, s=50, c=bond_lengths, cmap='viridis')
    # Add perfect prediction line
    min_energy = min(min(true_energies), min(predicted_energies))
    max_energy = max(max(true_energies), max(predicted_energies))
    axes[1, 0].plot([min_energy, max_energy], [min_energy, max_energy], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('True Energy (Ha)')
    axes[1, 0].set_ylabel('Predicted Energy (Ha)')
    axes[1, 0].set_title('Predicted vs True Energy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add colorbar for bond lengths
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Bond Length (Bohr)')
    
    # 4. Error distribution histogram
    axes[1, 1].hist(errors, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.6f} Ha')
    axes[1, 1].set_xlabel('Absolute Error (Ha)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dissociation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a comprehensive dissociation curve plot
    plt.figure(figsize=(12, 8))
    
    # Main dissociation curve
    plt.plot(bond_lengths, true_energies, 'b-', linewidth=3, marker='o', markersize=8, 
             label='Reference Energy', zorder=3)
    plt.plot(bond_lengths, predicted_energies, 'r--', linewidth=3, marker='s', markersize=8, 
             label='NN-XC Predicted Energy', zorder=2)
    
    # Add error bars or shaded region
    error_array = np.array(errors)
    plt.fill_between(bond_lengths, 
                     np.array(predicted_energies) - error_array,
                     np.array(predicted_energies) + error_array,
                     alpha=0.2, color='red', label='Prediction Error Range')
    
    plt.xlabel('H-H Bond Length (Bohr)', fontsize=14)
    plt.ylabel('Total Energy (Ha)', fontsize=14)
    plt.title('H₂ Molecule Dissociation Curve\nNeural Network XC Functional', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f"""
    Model Performance:
    • MAE: {np.mean(errors):.6f} Ha
    • RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.6f} Ha
    • Max Error: {np.max(errors):.6f} Ha
    • Min Error: {np.min(errors):.6f} Ha
    • Training Points: {len(bond_lengths)}
    """
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'h2_dissociation_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_rate_schedule(learning_rates, save_path):
    """Plot the learning rate schedule used during training"""
    epochs = list(range(len(learning_rates)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule During Training')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
    plt.close()

######## Training loop ########
writer = SummaryWriter()
initepoch = 0
num_epochs = 1001
lr = 1e-4

# For recording training history
training_history = []
learning_rates_history = []

for epoch in range(initepoch + 1, num_epochs + initepoch + 1):
    # Record learning rate
    learning_rates_history.append(lr)
    
    # Run an optimization step over a training batch
    state = params, opt_state, cost_val
    state, batch_metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state
    
    # Record training history
    training_history.append(epoch_metrics)
    
    # Print current epoch metrics
    print(f"\nEpoch {epoch} completed:")
    print(f"  Loss: {epoch_metrics.get('cost_value', 0):.6f}")
    print(f"  MAE: {epoch_metrics.get('mean_abs_error', 0):.6f} Ha")
    print(f"  MSE: {epoch_metrics.get('mean_sq_error', 0):.6f} Ha²")
    
    if epoch % 10 == 0:  # Save checkpoint every 10 epochs
        save_checkpoint_safe(params, opt_state, epoch, ckpt_dir)
        # Save training history
        with open(os.path.join(metrics_dir, f'training_history_epoch_{epoch}.pkl'), 'wb') as f:
            pickle.dump(training_history, f)
        
        # Plot current training curves
        plot_training_curves(training_history, metrics_dir)
    print("-" * 50)

# Second training phase
initepoch = 1001
num_epochs = 1000
lr = 1e-5
tx = adam(learning_rate=lr, b1=momentum)

for epoch in range(initepoch + 1, num_epochs + initepoch + 1):
    learning_rates_history.append(lr)
    
    state = params, opt_state, cost_val
    state, batch_metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state
    
    # Record training history
    training_history.append(epoch_metrics)
    
    # Print current epoch metrics
    print(f"\nEpoch {epoch} completed:")
    print(f"  Loss: {epoch_metrics.get('cost_value', 0):.6f}")
    print(f"  MAE: {epoch_metrics.get('mean_abs_error', 0):.6f} Ha")
    
    if epoch % 10 == 0:  # Save checkpoint every 10 epochs
        save_checkpoint_safe(params, opt_state, epoch, ckpt_dir)
        # Save training history
        with open(os.path.join(metrics_dir, f'training_history_epoch_{epoch}.pkl'), 'wb') as f:
            pickle.dump(training_history, f)
        
        # Plot current training curves
        plot_training_curves(training_history, metrics_dir)
    print("-" * 50)

# Third training phase
initepoch = 2001
num_epochs = 1000
lr = 1e-6
tx = adam(learning_rate=lr, b1=momentum)

for epoch in range(initepoch + 1, num_epochs + initepoch + 1):
    learning_rates_history.append(lr)
    
    state = params, opt_state, cost_val
    state, batch_metrics, epoch_metrics = train_epoch(state, training_files, training_data_dirpath)
    params, opt_state, cost_val = state
    
    # Record training history
    training_history.append(epoch_metrics)
    
    # Print current epoch metrics
    print(f"\nEpoch {epoch} completed:")
    print(f"  Loss: {epoch_metrics.get('cost_value', 0):.6f}")
    print(f"  MAE: {epoch_metrics.get('mean_abs_error', 0):.6f} Ha")
    
    if epoch % 10 == 0:  # Save checkpoint every 10 epochs
        save_checkpoint_safe(params, opt_state, epoch, ckpt_dir)
        # Save training history
        with open(os.path.join(metrics_dir, f'training_history_epoch_{epoch}.pkl'), 'wb') as f:
            pickle.dump(training_history, f)
        
        # Plot current training curves
        plot_training_curves(training_history, metrics_dir)
    print("-" * 50)

print("Training completed!")

# Final evaluation of the model on training set
print("\nEvaluating final model on training set...")
final_metrics, all_results = evaluate_model(params, training_files, training_data_dirpath)

if final_metrics:
    print("\nFinal model performance on training set:")
    print(f"  Mean Absolute Error (MAE): {final_metrics['mean_absolute_error']:.6f} Ha")
    print(f"  Root Mean Squared Error (RMSE): {final_metrics['root_mean_squared_error']:.6f} Ha")
    print(f"  Max Absolute Error: {final_metrics['max_absolute_error']:.6f} Ha")
    print(f"  Min Absolute Error: {final_metrics['min_absolute_error']:.6f} Ha")
    print(f"  Total samples: {final_metrics['num_samples']}")
    
    # Plot dissociation curve analysis
    plot_dissociation_curve(all_results, metrics_dir)
    
    # Plot learning rate schedule
    plot_learning_rate_schedule(learning_rates_history, metrics_dir)
    
    # Save final evaluation metrics
    with open(os.path.join(metrics_dir, 'final_evaluation_metrics.json'), 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.ndarray, np.generic)) else v 
                  for k, v in final_metrics.items()}, f, indent=4)
    
    # Save detailed results
    detailed_results = []
    for r in all_results:
        detailed_results.append({
            'bond_length': float(r['bond_length']) if r['bond_length'] is not None else None,
            'true_energy': float(r['true_energy']),
            'predicted_energy': float(r['predicted_energy']),
            'absolute_error': float(r['absolute_error'])
        })
    
    # Save as CSV if pandas is available
    try:
        import pandas as pd
        df = pd.DataFrame(detailed_results)
        df.to_csv(os.path.join(metrics_dir, 'detailed_predictions.csv'), index=False)
        print(f"\nDetailed predictions saved to: {os.path.join(metrics_dir, 'detailed_predictions.csv')}")
    except ImportError:
        # Save as numpy text file
        np.savetxt(os.path.join(metrics_dir, 'detailed_predictions.txt'), 
                  np.array([(r['bond_length'] or -1, r['true_energy'], r['predicted_energy'], r['absolute_error']) 
                           for r in detailed_results]),
                  header='bond_length true_energy predicted_energy absolute_error',
                  fmt='%.6f')
        print(f"\nDetailed predictions saved to: {os.path.join(metrics_dir, 'detailed_predictions.txt')}")

# Plot complete training curves
print("\nPlotting complete training curves...")
plot_training_curves(training_history, metrics_dir)

# Save complete training history
with open(os.path.join(metrics_dir, 'complete_training_history.pkl'), 'wb') as f:
    pickle.dump(training_history, f)

print(f"\nAll training metrics and plots saved to: {metrics_dir}")
print("Training and evaluation completed successfully!")