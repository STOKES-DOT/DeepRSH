import os

os.environ["TYPEGUARD_TYPECHECKER"] = "None" 
os.environ["JAXTYPING_DISABLE"] = "1" 
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.random import PRNGKey, split
from jax.nn import gelu
from functools import partial
from optax import adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from orbax.checkpoint import PyTreeCheckpointer
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import jax
import numpy as np
import pickle
from grad_dft import (
    train_kernel, 
    energy_predictor,
    DispersionFunctional,
    NeuralFunctional,
    canonicalize_inputs,
    dm21_coefficient_inputs,
    densities,
    dm21_combine_cinputs,
    dm21_combine_densities,
    dm21_hfgrads_cinputs,
    dm21_hfgrads_densities,
    loader
)
from xc_functional import rsh_b3lyp_nn
from jax import config
config.update("jax_enable_x64", True)

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    "SEED": 42,
    "MOMENTUM": 0.9,
    "DATA_PATH": "h2_dataset_n.hdf5",  # Ensure this file exists
    "CKPT_DIR_BASE": "/home/yjiao/DeepRSH/module/ckpts/Standard_Training",
    "LOG_DIR": "/home/yjiao/DeepRSH/module/runs/Standard_Training",
    "PLOT_DIR": "/home/yjiao/DeepRSH/module/plots/Standard_Training"
}

# Ensure directories exist
os.makedirs(CONFIG["CKPT_DIR_BASE"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
os.makedirs(CONFIG["PLOT_DIR"], exist_ok=True)

# ==========================================
# Helper Functions
# ==========================================

def convert(o):
    """JSON converter for numpy types"""
    if isinstance(o, np.float32):
        return float(o)
    return o

def safe_save_checkpoints(functional, params, tx, step, orbax_checkpointer, ckpt_dir):
    """安全保存检查点，处理 JAX 设备数组转换"""

    
    # 确保目录存在
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 使用 Flax 的 checkpoints 模块
    from flax.training import checkpoints
    
    try:
        # 只保存参数，不保存优化器状态
        # 将参数转换为纯 Python/NumPy 结构
        params_numpy = jax.tree_map(lambda x: np.array(x), params)
        
        # 创建要保存的字典
        target = {
            'params': params_numpy,
            'step': step,
            'config': CONFIG
        }
        
        # 使用 Flax 的 save_checkpoint
        checkpoint_path = checkpoints.save_checkpoint(
            ckpt_dir=ckpt_dir,
            target=target,
            step=step,
            keep=1,  # 只保留最新的
            overwrite=True
        )
        
        print(f"Successfully saved checkpoint at step {step} to {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Failed to save checkpoint at step {step}: {e}")
        
        # 尝试使用 pickle 保存
        try:
            # 将参数转换为主机内存
            params_host = jax.device_get(params)
            
            # 创建保存字典
            save_dict = {
                'params': params_host,
                'step': step,
                'config': CONFIG
            }
            
            # 使用 pickle 保存
            checkpoint_path = os.path.join(ckpt_dir, f'checkpoint_{step}.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(save_dict, f)
            
            print(f"Saved checkpoint using pickle format at {checkpoint_path}")
            return True
        except Exception as e2:
            print(f"Also failed with pickle save: {e2}")
            
            # 最后尝试：使用 numpy 保存参数权重
            try:
                # 展平并保存参数
                params_flat, params_tree = jax.tree_util.tree_flatten(params)
                params_arrays = [np.array(x) for x in params_flat]
                
                # 保存每个数组
                for i, arr in enumerate(params_arrays):
                    array_path = os.path.join(ckpt_dir, f'checkpoint_{step}_param_{i}.npy')
                    np.save(array_path, arr)
                
                # 保存树结构
                tree_path = os.path.join(ckpt_dir, f'checkpoint_{step}_tree.pkl')
                with open(tree_path, 'wb') as f:
                    pickle.dump(params_tree, f)
                
                # 保存元数据
                meta_path = os.path.join(ckpt_dir, f'checkpoint_{step}_meta.pkl')
                with open(meta_path, 'wb') as f:
                    pickle.dump({
                        'step': step,
                        'num_params': len(params_arrays),
                        'config': CONFIG
                    }, f)
                
                print(f"Saved checkpoint as separate numpy files at {ckpt_dir}")
                return True
            except Exception as e3:
                print(f"All save attempts failed: {e3}")
                return False

def plot_loss_curves(epoch_results, plot_dir):
    """Plot loss curves and save to file"""
    if not epoch_results:
        print("No epoch results to plot.")
        return
    
    epochs = sorted(epoch_results.keys())
    
    # Extract metrics
    mse_values = [epoch_results[e]['mean_sq_error'] for e in epochs]
    mae_values = [epoch_results[e]['mean_abs_error'] for e in epochs]
    cost_values = [epoch_results[e]['cost_value'] for e in epochs]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Loss Curves (Log Scale)', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean Squared Error (log scale)
    ax1 = axes[0, 0]
    ax1.semilogy(epochs, mse_values, 'b-', linewidth=2, label='MSE')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Squared Error (log scale)', fontsize=12)
    ax1.set_title('Mean Squared Error (MSE)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add a horizontal line at the minimum value
    min_mse = min(mse_values)
    min_epoch = epochs[mse_values.index(min_mse)]
    ax1.axhline(y=min_mse, color='r', linestyle='--', alpha=0.5, 
                label=f'Min MSE: {min_mse:.2e} at epoch {min_epoch}')
    ax1.legend(fontsize=10)
    
    # Plot 2: Mean Absolute Error (log scale)
    ax2 = axes[0, 1]
    ax2.semilogy(epochs, mae_values, 'g-', linewidth=2, label='MAE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (log scale)', fontsize=12)
    ax2.set_title('Mean Absolute Error (MAE)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Add a horizontal line at the minimum value
    min_mae = min(mae_values)
    min_epoch_mae = epochs[mae_values.index(min_mae)]
    ax2.axhline(y=min_mae, color='r', linestyle='--', alpha=0.5,
                label=f'Min MAE: {min_mae:.2e} at epoch {min_epoch_mae}')
    ax2.legend(fontsize=10)
    
    # Plot 3: Cost Value (log scale)
    ax3 = axes[1, 0]
    ax3.semilogy(epochs, cost_values, 'r-', linewidth=2, label='Cost')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Cost Value (log scale)', fontsize=12)
    ax3.set_title('Cost Function Value', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    
    # Add a horizontal line at the minimum value
    min_cost = min(cost_values)
    min_epoch_cost = epochs[cost_values.index(min_cost)]
    ax3.axhline(y=min_cost, color='b', linestyle='--', alpha=0.5,
                label=f'Min Cost: {min_cost:.2e} at epoch {min_epoch_cost}')
    ax3.legend(fontsize=10)
    
    # Plot 4: Combined plot (all metrics)
    ax4 = axes[1, 1]
    ax4.semilogy(epochs, mse_values, 'b-', linewidth=2, label='MSE', alpha=0.7)
    ax4.semilogy(epochs, mae_values, 'g-', linewidth=2, label='MAE', alpha=0.7)
    ax4.semilogy(epochs, cost_values, 'r-', linewidth=2, label='Cost', alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Error (log scale)', fontsize=12)
    ax4.set_title('Combined Loss Curves', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=12)
    
    # Add learning rate schedule annotations if available
    if 'learning_rate' in epoch_results[epochs[0]]:
        lr_values = [epoch_results[e]['learning_rate'] for e in epochs]
        unique_lrs = sorted(set(lr_values))
        
        # Add vertical lines for LR changes
        for lr in unique_lrs:
            lr_epochs = [e for e in epochs if epoch_results[e]['learning_rate'] == lr]
            if lr_epochs:
                ax4.axvline(x=min(lr_epochs), color='gray', linestyle=':', alpha=0.5)
                ax4.text(min(lr_epochs), ax4.get_ylim()[1]*0.1, f'LR={lr:.1e}', 
                        rotation=90, fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to {plot_path}")
    
    # Also save as PDF for publication quality
    plot_path_pdf = os.path.join(plot_dir, 'loss_curves.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    
    plt.close()
    
    # Create a separate detailed log-log plot
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Use log-log scale for more detailed view
    ax.loglog(epochs, mse_values, 'b-', linewidth=3, label='MSE', alpha=0.8)
    ax.loglog(epochs, mae_values, 'g--', linewidth=3, label='MAE', alpha=0.8)
    ax.loglog(epochs, cost_values, 'r-.', linewidth=3, label='Cost', alpha=0.8)
    
    ax.set_xlabel('Epoch (log scale)', fontsize=14)
    ax.set_ylabel('Error (log scale)', fontsize=14)
    ax.set_title('Training Convergence (Log-Log Scale)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)
    
    # Add exponential fit lines for last 50 epochs
    if len(epochs) > 50:
        last_epochs = epochs[-50:]
        last_mse = mse_values[-50:]
        
        # Fit exponential decay
        try:
            log_mse = np.log(last_mse)
            coeffs = np.polyfit(last_epochs, log_mse, 1)
            fit_mse = np.exp(coeffs[1] + coeffs[0] * np.array(last_epochs))
            
            ax.loglog(last_epochs, fit_mse, 'k:', linewidth=2, 
                     label=f'Exp fit: y∝exp({coeffs[0]:.3f}x)', alpha=0.7)
            ax.legend(fontsize=12)
        except:
            pass
    
    plt.tight_layout()
    
    # Save detailed plot
    detailed_plot_path = os.path.join(plot_dir, 'loss_curves_detailed.png')
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed loss curves saved to {detailed_plot_path}")
    
    # Create a summary statistics file
    stats_path = os.path.join(plot_dir, 'training_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Training Statistics Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Epochs: {len(epochs)}\n")
        f.write(f"Final Epoch: {epochs[-1]}\n\n")
        
        f.write("Minimum Values:\n")
        f.write(f"  MSE:  {min_mse:.6e} (epoch {min_epoch})\n")
        f.write(f"  MAE:  {min_mae:.6e} (epoch {min_epoch_mae})\n")
        f.write(f"  Cost: {min_cost:.6e} (epoch {min_epoch_cost})\n\n")
        
        f.write("Final Values:\n")
        f.write(f"  MSE:  {mse_values[-1]:.6e}\n")
        f.write(f"  MAE:  {mae_values[-1]:.6e}\n")
        f.write(f"  Cost: {cost_values[-1]:.6e}\n\n")
        
        if len(epochs) > 1:
            f.write("Convergence Analysis:\n")
            mse_improvement = (mse_values[0] - mse_values[-1]) / mse_values[0] * 100
            f.write(f"  MSE Improvement: {mse_improvement:.2f}%\n")
            
            # Calculate average epoch-to-epoch improvement
            mse_changes = [(mse_values[i] - mse_values[i+1])/mse_values[i] 
                          for i in range(len(mse_values)-1) if mse_values[i] > 0]
            if mse_changes:
                avg_improvement = np.mean(mse_changes) * 100
                f.write(f"  Average Epoch Improvement: {avg_improvement:.4f}%\n")
    
    print(f"Training statistics saved to {stats_path}")

# Initialize Functional
n_layers = 10
width_layers = 512
squash_offset = 1e-4
layer_widths = [width_layers] * n_layers
nlc_layer_widths = [width_layers // 4] * (n_layers // 2)
out_features = 20  # 2 for each spin x 2 for exchange/correlation x 4 for MGGA + 4 for HF
sigmoid_scale_factor = 2.0
activation = gelu
loadcheckpoint = False


def nn_coefficients(instance, rhoinputs, *_, **__):
    x = canonicalize_inputs(rhoinputs)  # Making sure dimensions are correct

    # Initial layer: log -> dense -> tanh
    x = jnp.log(jnp.abs(x) + squash_offset)  # squash_offset = 1e-4
    instance.sow("intermediates", "log", x)
    x = instance.dense(features=layer_widths[0])(x)  # features = 256
    instance.sow("intermediates", "initial_dense", x)
    x = jnp.tanh(x)
    instance.sow("intermediates", "tanh", x)

    # 6 Residual blocks with 256-features dense layer and layer norm
    for features, i in zip(layer_widths, range(len(layer_widths))):  # layer_widths = [256]*6
        res = x
        x = instance.dense(features=features)(x)
        instance.sow("intermediates", "residual_dense_" + str(i), x)
        x = x + res  # nn.Dense + Residual connection
        instance.sow("intermediates", "residual_residual_" + str(i), x)
        x = instance.layer_norm()(x)  # + res # nn.LayerNorm
        instance.sow("intermediates", "residual_layernorm_" + str(i), x)
        x = activation(x)  # activation = jax.nn.gelu
        instance.sow("intermediates", "residual_elu_" + str(i), x)

    return instance.head(x, out_features, sigmoid_scale_factor)

def combine_densities(densities, ehf):
    ehf = jnp.reshape(ehf, (ehf.shape[2], ehf.shape[0] * ehf.shape[1]))
    return jnp.concatenate((densities, ehf), axis=1)

omegas = jnp.array([0.0, 0.3])
functional = NeuralFunctional(
    coefficients=nn_coefficients,
    energy_densities=partial(densities, functional_type="MGGA"),
    nograd_densities=lambda molecule, *_, **__: molecule.HF_energy_density(omegas),
    densitygrads=lambda self, params, molecule, nograd_densities, cinputs, grad_densities, *_, **__: dm21_hfgrads_densities(
        self, params, molecule, nograd_densities, cinputs, grad_densities, omegas
    ),
    combine_densities=combine_densities,
    coefficient_inputs=dm21_coefficient_inputs,
    nograd_coefficient_inputs=lambda molecule, *_, **__: molecule.HF_energy_density(omegas),
    coefficient_input_grads=lambda self, params, molecule, nograd_cinputs, grad_cinputs, densities, *_, **__: dm21_hfgrads_cinputs(
        self, params, molecule, nograd_cinputs, grad_cinputs, densities, omegas
    ),
    combine_inputs=dm21_combine_cinputs,
)

# Predictor
compute_energy = energy_predictor(functional)

@partial(value_and_grad, has_aux=True)
def loss(params, molecule, true_energy):
    predicted_energy, fock = compute_energy(params, molecule)
    cost_value = (predicted_energy - true_energy) ** 2
    
    metrics = {
        "mean_abs_error": jnp.mean(jnp.abs(predicted_energy - true_energy)),
        "mean_sq_error": jnp.mean((predicted_energy - true_energy) ** 2),
        "cost_value": cost_value,
        "predicted_energy": predicted_energy,
        "ground_truth_energy": true_energy
    }
    return cost_value, metrics

def train_epoch(state, kernel, dataset):
    """Train for a single epoch using the GradDFT kernel."""
    batch_metrics = []
    params, opt_state, cost_val = state

    # Iterate over molecules directly
    # Note: grad_dft loader returns a generator, so we iterate over the list passed in
    i=1
    for system in tqdm(dataset, desc="Molecules"):
        if i ==1:
            print(system[1])
        # The kernel handles the update: params, opt_state, cost, metrics
        params, opt_state, cost_val, metrics = kernel(params, opt_state, system[1], system[1].energy)
        
        # Convert JAX metrics to python for logging
        batch_metrics.append(metrics)

    # Average metrics for the epoch
    epoch_metrics = {
        k: np.mean([jax.device_get(m[k]) for m in batch_metrics])
        for k in batch_metrics[0]
    }
    
    state = (params, opt_state, cost_val)
    return state, metrics, epoch_metrics

# ==========================================
# Main Training Logic
# ==========================================

def main():
    writer = SummaryWriter(log_dir=CONFIG["LOG_DIR"])
    orbax_checkpointer = PyTreeCheckpointer()
    
    # 1. Initialize Parameters
    key = PRNGKey(CONFIG["SEED"])
    key, subkey = split(key)
    rhoinputs = jax.random.normal(key, shape=[2, 7])
    params = functional.init(subkey, rhoinputs)
    
    # 2. Load Data
    print(f"Loading data from {CONFIG['DATA_PATH']}...")
    # Load all data into a list to reuse across epochs
    omegas = jnp.array([0.0, 0.3])
    dataset = list(loader(fname=CONFIG['DATA_PATH'], randomize=True, training=True, config_omegas=omegas))
    print(f"Loaded {len(dataset)} molecules.")

    # 3. Define Schedule (Epochs, LR)
    # Format: (num_epochs, learning_rate)
    schedule = [
        (200, 1e-4),
        (200, 1e-5),
        (200, 1e-6),
        (100,  1e-7),
    ]

    current_epoch = 0
    epoch_results = {}
    results_path_json = os.path.join(CONFIG["LOG_DIR"], 'epoch_results.json')

    # 4. Training Loop
    # We iterate through the schedule stages, re-initializing the optimizer 
    # and kernel for each learning rate change, but keeping the params.
    
    cost_val = jnp.inf

    for stage_idx, (stage_epochs, lr) in enumerate(schedule):
        print(f"\n=== Starting Stage {stage_idx+1}: LR={lr}, Epochs={stage_epochs} ===")
        
        # Re-initialize optimizer with new LR
        tx = adam(learning_rate=lr, b1=CONFIG["MOMENTUM"])
        opt_state = tx.init(params)
        
        # JIT compile the kernel with the new optimizer
        kernel = jax.jit(train_kernel(tx, loss))
        
        state = (params, opt_state, cost_val)

        for _ in range(stage_epochs):
            current_epoch += 1
            
            # Run one epoch
            state, _, epoch_metrics = train_epoch(state, kernel, dataset)
            params, opt_state, cost_val = state

            # Add learning rate to metrics for plotting
            epoch_metrics['learning_rate'] = lr
            
            # Logging
            epoch_results[current_epoch] = epoch_metrics
            print(f"Epoch {current_epoch} (LR {lr}): MSE={epoch_metrics['mean_sq_error']:.6e}, MAE={epoch_metrics['mean_abs_error']:.6e}, Cost={epoch_metrics['cost_value']:.6e}")
            
            # Log all metrics to TensorBoard
            for k, v in epoch_metrics.items():
                writer.add_scalar(f"Train/{k}", v, current_epoch)
            
            # Also log learning rate separately
            writer.add_scalar("Hyperparameters/learning_rate", lr, current_epoch)
            
            # Add histogram of parameters periodically
            if current_epoch % 50 == 0:
                # Log parameter histograms
                for param_name, param_value in jax.tree_util.tree_flatten_with_path(params)[0]:
                    # Convert path to string
                    path_str = '/'.join(str(p.key) for p in param_name)
                    writer.add_histogram(f"Parameters/{path_str}", 
                                        jax.device_get(param_value), 
                                        current_epoch)
            
            writer.flush()
            
            # Checkpointing every 20 epochs or at end of stage
            if current_epoch % 20 == 0:
                safe_save_checkpoints(
                    functional, params, tx,
                    step=current_epoch,
                    orbax_checkpointer=orbax_checkpointer,
                    ckpt_dir=CONFIG["CKPT_DIR_BASE"]
                )
                
                # Save JSON logs
                with open(results_path_json, 'w') as fp:
                    json.dump(epoch_results, fp, default=convert)
                
                # Plot loss curves at each checkpoint
                plot_loss_curves(epoch_results, CONFIG["PLOT_DIR"])

    # Final checkpoint
    print("Saving final checkpoint...")
    safe_save_checkpoints(
        functional, params, tx,
        step=current_epoch,
        orbax_checkpointer=orbax_checkpointer,
        ckpt_dir=CONFIG["CKPT_DIR_BASE"]
    )
    
    # Final JSON save
    with open(results_path_json, 'w') as fp:
        json.dump(epoch_results, fp, default=convert)
    
    # Final plot
    plot_loss_curves(epoch_results, CONFIG["PLOT_DIR"])
    
    # Create convergence analysis
    print("\n" + "="*60)
    print("Training Complete - Convergence Analysis")
    print("="*60)
    
    epochs = sorted(epoch_results.keys())
    if epochs:
        first_epoch = epochs[0]
        last_epoch = epochs[-1]
        
        initial_mse = epoch_results[first_epoch]['mean_sq_error']
        final_mse = epoch_results[last_epoch]['mean_sq_error']
        improvement = (initial_mse - final_mse) / initial_mse * 100
        
        print(f"Total Epochs: {len(epochs)}")
        print(f"Initial MSE (epoch {first_epoch}): {initial_mse:.6e}")
        print(f"Final MSE (epoch {last_epoch}): {final_mse:.6e}")
        print(f"Improvement: {improvement:.2f}%")
        print(f"Final model saved to {CONFIG['CKPT_DIR_BASE']}")
        print(f"Loss curves saved to {CONFIG['PLOT_DIR']}")
    
    writer.close()
    
    # Generate a final comprehensive report
    generate_training_report(epoch_results, CONFIG)

def generate_training_report(epoch_results, config):
    """Generate a comprehensive training report"""
    report_path = os.path.join(config["PLOT_DIR"], "training_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DEEP RSH NEURAL FUNCTIONAL - TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-"*30 + "\n")
        f.write(f"Data Path: {config['DATA_PATH']}\n")
        f.write(f"Seed: {config['SEED']}\n")
        f.write(f"Momentum: {config['MOMENTUM']}\n")
        f.write(f"Checkpoint Directory: {config['CKPT_DIR_BASE']}\n")
        f.write(f"Log Directory: {config['LOG_DIR']}\n")
        f.write(f"Plot Directory: {config['PLOT_DIR']}\n\n")
        
        f.write("TRAINING SUMMARY:\n")
        f.write("-"*30 + "\n")
        
        epochs = sorted(epoch_results.keys())
        if not epochs:
            f.write("No training data available.\n")
            return
        
        f.write(f"Total Epochs: {len(epochs)}\n")
        f.write(f"Training Range: Epoch {min(epochs)} to {max(epochs)}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*30 + "\n")
        
        # Find best epochs for each metric
        best_mse_epoch = min(epochs, key=lambda e: epoch_results[e]['mean_sq_error'])
        best_mae_epoch = min(epochs, key=lambda e: epoch_results[e]['mean_abs_error'])
        best_cost_epoch = min(epochs, key=lambda e: epoch_results[e]['cost_value'])
        
        f.write(f"Best MSE:  {epoch_results[best_mse_epoch]['mean_sq_error']:.6e} (epoch {best_mse_epoch})\n")
        f.write(f"Best MAE:  {epoch_results[best_mae_epoch]['mean_abs_error']:.6e} (epoch {best_mae_epoch})\n")
        f.write(f"Best Cost: {epoch_results[best_cost_epoch]['cost_value']:.6e} (epoch {best_cost_epoch})\n\n")
        
        f.write("FINAL EPOCH METRICS:\n")
        f.write("-"*30 + "\n")
        last_epoch = max(epochs)
        last_metrics = epoch_results[last_epoch]
        
        for metric, value in last_metrics.items():
            if metric != 'learning_rate':
                f.write(f"{metric.replace('_', ' ').title()}: {value:.6e}\n")
        
        f.write(f"\nFinal Learning Rate: {last_metrics.get('learning_rate', 'N/A')}\n")
        
        f.write("\nCONVERGENCE ANALYSIS:\n")
        f.write("-"*30 + "\n")
        
        first_mse = epoch_results[min(epochs)]['mean_sq_error']
        last_mse = epoch_results[max(epochs)]['mean_sq_error']
        
        if first_mse > 0:
            convergence = (first_mse - last_mse) / first_mse * 100
            f.write(f"MSE Convergence: {convergence:.2f}% reduction\n")
        
        # Calculate average epoch-to-epoch improvement
        mse_values = [epoch_results[e]['mean_sq_error'] for e in epochs]
        improvements = []
        for i in range(1, len(mse_values)):
            if mse_values[i-1] > 0:
                improvement = (mse_values[i-1] - mse_values[i]) / mse_values[i-1] * 100
                improvements.append(improvement)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            f.write(f"Average Epoch Improvement: {avg_improvement:.4f}%\n")
            f.write(f"Standard Deviation: {np.std(improvements):.4f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"Training report saved to {report_path}")

if __name__ == "__main__":
    main()