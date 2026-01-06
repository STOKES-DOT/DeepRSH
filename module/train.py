import os
import pickle

from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.random import PRNGKey, split
from optax import adam
from jax.nn import gelu
from jaxtyping import Array, Float, PyTree, Scalar

import grad_dft as gd
from grad_dft import (
    energy_predictor,
    NeuralFunctional,
    Molecule,
    loader,
    canonicalize_inputs,
    correlation_polarization_correction,
    exchange_polarization_correction,
)

# ==========================================
# 1. 物理泛函成分定义 (从您的代码片段整合)
# ==========================================

def lsda_x_e(rho: Float[Array, "grid spin"], clip_cte) -> Float[Array, "grid"]:
    rho = jnp.clip(rho, a_min=clip_cte)
    lda_es = (
        -3.0
        / 4.0
        * (jnp.array([[3.0, 6.0]]) / jnp.pi) ** (1 / 3)
        * (rho.sum(axis=1, keepdims=True)) ** (4 / 3)
    )
    return exchange_polarization_correction(lda_es, rho)

def b88_x_e(rho: Float[Array, "grid spin"], grad_rho: Float[Array, "grid spin dimension"], clip_cte: float = 1e-30) -> Float[Array, "grid"]:
    beta = 0.0042
    rho = jnp.clip(rho, a_min=clip_cte)
    log_rho = jnp.log2(jnp.clip(rho, a_min=clip_cte))
    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)
    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min=clip_cte)) / 2
    log_x_sigma = log_grad_rho_norm - 4 / 3.0 * log_rho
    x_sigma = 2**log_x_sigma
    return -(
        beta
        * 2
        ** (
            4 * log_rho / 3
            + 2 * log_x_sigma
            - jnp.log2(1 + 6 * beta * x_sigma * jnp.arcsinh(x_sigma))
        )
    ).sum(axis=1)

def vwn_c_e(rho: Float[Array, "grid spin"], clip_cte: float = 1e-30) -> Float[Array, "grid"]:
    A = jnp.array([[0.0621814, 0.0621814 / 2]])
    b = jnp.array([[3.72744, 7.06042]])
    c = jnp.array([[12.9352, 18.0578]])
    x0 = jnp.array([[-0.10498, -0.325]])

    rho = jnp.where(rho > clip_cte, rho, 0.0)
    log_rho = jnp.log2(jnp.clip(rho.sum(axis=1, keepdims=True), a_min=clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_rho / 3.0
    log_x = log_rs / 2
    rs = 2**log_rs
    x = 2**log_x

    X = 2 ** (2 * log_x) + 2 ** (log_x + jnp.log2(b)) + c
    X0 = x0**2 + b * x0 + c
    Q = jnp.sqrt(4 * c - b**2)

    e_PF = (
        A / 2 * (
            2 * jnp.log(x) - jnp.log(X) + 2 * b / Q * jnp.arctan(Q / (2 * x + b))
            - b * x0 / X0 * (jnp.log((x - x0) ** 2 / X) + 2 * (2 * x0 + b) / Q * jnp.arctan(Q / (2 * x + b)))
        )
    )
    e_tilde = correlation_polarization_correction(e_PF, rho, clip_cte)
    return e_tilde * rho.sum(axis = 1)

def lyp_c_e(rho: Float[Array, "grid spin"], grad_rho: Float[Array, "grid spin 3"], grad2rho: Float[Array, "grid spin"], clip_cte=1e-30) -> Float[Array, "grid"]:
    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    CF = (3 / 10) * (3 * jnp.pi**2) ** (2 / 3)

    rho = jnp.clip(rho, a_min=clip_cte)
    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)
    t = (jnp.where(rho > clip_cte, grad_rho_norm_sq / rho, 0) - grad2rho) / 8.0
    frac = jnp.where(rho.sum(axis=1) > clip_cte, ((rho**2).sum(axis=1)) / (rho.sum(axis=1)) ** 2, 1)
    gamma = 2 * (1 - frac)
    rhos_ts = rho.sum(axis=1) * t.sum(axis=1)
    rho_t = (rho * t).sum(axis=1)
    rho_grad2rho = (rho * grad2rho).sum(axis=1)
    rhom1_3 = (rho.sum(axis=1)) ** (-1 / 3)
    rho8_3 = (rho ** (8 / 3)).sum(axis=1)
    rhom5_3 = (rho.sum(axis=1)) ** (-5 / 3)
    exp_factor = jnp.where(rho.sum(axis=1) > 0, jnp.exp(-c * rhom1_3), 0)
    parenthesis = 2 ** (2 / 3) * CF * (rho8_3) - rhos_ts + rho_t / 9 + rho_grad2rho / 18
    braket_m_rho = jnp.where(rho.sum(axis=1) > clip_cte, 2 * b * rhom5_3 * parenthesis * exp_factor, 0.0)

    return -a * jnp.where(
        rho.sum(axis=1) > clip_cte, gamma / (1 + d * rhom1_3) * (rho.sum(axis=1) + braket_m_rho), 0.0
    )

# ==========================================
# 2. 神经网络泛函定义
# ==========================================

squash_offset = 1e-4
layer_widths = [256] * 6
out_features = 6 
sigmoid_scale_factor = 2.0
activation = gelu

def nn_coefficients(instance, rhoinputs, *_, **__):
    x = canonicalize_inputs(rhoinputs)
    x = jnp.log(jnp.abs(x) + squash_offset)
    x = instance.dense(features=layer_widths[0])(x)
    x = jnp.tanh(x)

    for features in layer_widths:
        res = x
        x = instance.dense(features=features)(x)
        x = x + res
        x = instance.layer_norm()(x)
        x = activation(x)

    return instance.head(x, out_features, sigmoid_scale_factor)

def energy_density_rsh(molecule: Molecule, clip_cte: float = 1e-30, *_, **__):
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    grad2rho = molecule.lapl_density()
    
    lda_e = lsda_x_e(rho, clip_cte)
    b88_e = b88_x_e(rho, grad_rho, clip_cte)
    vwn_e = vwn_c_e(rho, clip_cte)
    lyp_e = lyp_c_e(rho, grad_rho, grad2rho, clip_cte)

    OMEGA = 0.4
    hf_energies = molecule.HF_energy_density([0.0, OMEGA])
    hf_full = hf_energies[:, 0]
    hf_lr = hf_energies[:, 1]
    hf_sr = hf_full - hf_lr

    return jnp.stack((lda_e, b88_e, vwn_e, lyp_e, hf_sr, hf_lr), axis=1)

def rsh_b3lyp_nn():
    return NeuralFunctional(
        coefficients=nn_coefficients,
        energy_densities=energy_density_rsh,
        coefficient_inputs=gd.dm21_coefficient_inputs,
    )

# ==========================================
# 3. 损失函数 (Janak Loss)
# ==========================================

def get_homo_energy(fock: Array, molecule: Molecule) -> Scalar:
    w, _ = jax.scipy.linalg.eigh(fock, b=molecule.s1e)
    n_occ = jnp.sum(molecule.mo_occ > 1e-1).astype(int)
    homo_index = n_occ - 1
    return w[homo_index]

def janak_loss(params, compute_energy, molecule, cation, anion):
    E_N, fock_N = compute_energy(params, molecule)
    E_Np1, fock_Np1 = compute_energy(params, anion)
    E_Nm1, _ = compute_energy(params, cation)

    homo_N = get_homo_energy(fock_N, molecule)
    homo_Np1 = get_homo_energy(fock_Np1, anion)

    J_N = jnp.abs(homo_N + E_Nm1 - E_N)
    J_Np1 = jnp.abs(homo_Np1 + E_N - E_Np1)
    return J_N**2 + J_Np1**2

def total_loss(params, compute_energy, molecule, cation, anion, ground_truth_energy, janak_weight=1.0):
    janak_val = janak_loss(params, compute_energy, molecule, cation, anion)
    pred_energy, _ = compute_energy(params, molecule)
    mse_val = (pred_energy - ground_truth_energy) ** 2
    total_val = mse_val + janak_weight * janak_val
    
    metrics = {
        "total_loss": total_val,
        "mse_loss": mse_val,
        "janak_loss": janak_val,
        "pred_energy": pred_energy
    }
    return total_val, metrics

# ==========================================
# 4. 数据加载与绘图
# ==========================================

def load_and_pair_data(fpath: str) -> List[Tuple[Molecule, Molecule, Molecule]]:
    print(f"Loading data from {fpath}...")
    # 必须确保 config_omegas 与预处理一致
    all_mols_gen = loader(fname=fpath, randomize=False, training=True, config_omegas=[])
    all_mols = list(all_mols_gen)
    
    total_len = len(all_mols)
    if total_len % 3 != 0:
        raise ValueError(f"Total molecules ({total_len}) is not divisible by 3. Data might be corrupted.")
        
    n_samples = total_len // 3
    print(f"Detected {n_samples} systems (assuming structure: [Neutral... | Cation... | Anion...])")
    
    neutrals = all_mols[0 : n_samples]
    cations  = all_mols[n_samples : 2*n_samples]
    anions   = all_mols[2*n_samples : 3*n_samples]
    
    dataset = []
    # 简单的完整性检查
    for i in range(n_samples):
        # 这里可以加一个简单的 grid 大小检查，确保是同一个分子
        if neutrals[i].grid.coords.shape != cations[i].grid.coords.shape:
             print(f"Warning: Grid mismatch at index {i}, check data order!")
        dataset.append((neutrals[i], cations[i], anions[i]))
            
    return dataset
def plot_loss_history(history, save_path="loss_history.png"):
    epochs = range(1, len(history['total']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['total'], label='Total Loss', linewidth=2)
    plt.plot(epochs, history['mse'], label='MSE Loss', linestyle='--')
    plt.plot(epochs, history['janak'], label='Janak Loss', linestyle=':')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss curve saved to {save_path}")

# ==========================================
# 5. 主训练函数
# ==========================================

def train():
    # 参数配置
    SEED = 42
    LR = 1e-4
    MOMENTUM = 0.9
    EPOCHS = 100
    JANAK_WEIGHT = 1.0
    DATA_PATH = "h2_dataset.h5"
    CKPT_DIR = "ckpts/RSH_Janak"
    
    # 初始化网络
    key = PRNGKey(SEED)
    functional = rsh_b3lyp_nn()
    compute_energy_fn = energy_predictor(functional)
    
    key, subkey = split(key)
    params = functional.init(subkey, jax.random.normal(subkey, (2, 7)))
    
    tx = adam(learning_rate=LR, b1=MOMENTUM)
    opt_state = tx.init(params)
    
    # 加载数据
    dataset = load_and_pair_data(DATA_PATH)

    
    split_idx = int(len(dataset) * 0.8)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")

    # 编译训练步
    @jax.jit
    def update_step(params, opt_state, neutral, cation, anion):
        loss_fn = partial(
            total_loss, 
            compute_energy=compute_energy_fn, 
            ground_truth_energy=neutral.energy, 
            janak_weight=JANAK_WEIGHT
        )
        (cost, metrics), grads = value_and_grad(loss_fn, has_aux=True)(
            params, neutral, cation, anion
        )
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = jax.optax.apply_updates(params, updates)
        return new_params, new_opt_state, cost, metrics

    # 训练循环
    loss_history = {'total': [], 'mse': [], 'janak': []}
    
    print("\nStarting Training...")
    for epoch in range(1, EPOCHS + 1):

        epoch_metrics = {'total': [], 'mse': [], 'janak': []}
        
        pbar = tqdm(train_set, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for triplet in pbar:
            neutral, cation, anion = triplet
            params, opt_state, cost, m = update_step(params, opt_state, neutral, cation, anion)
            
            epoch_metrics['total'].append(m['total_loss'])
            epoch_metrics['mse'].append(m['mse_loss'])
            epoch_metrics['janak'].append(m['janak_loss'])
            
            pbar.set_postfix({'loss': f"{cost:.4f}"})
        
        avg_total = jnp.mean(epoch_metrics['total'])
        avg_mse = jnp.mean(epoch_metrics['mse'])
        avg_janak = jnp.mean(epoch_metrics['janak'])
        
        loss_history['total'].append(avg_total)
        loss_history['mse'].append(avg_mse)
        loss_history['janak'].append(avg_janak)
        
        print(f"Epoch {epoch}: Total={avg_total:.5f} | MSE={avg_mse:.5f} | Janak={avg_janak:.5f}")

        if epoch % 10 == 0 and test_set:
            test_losses = []
            for triplet in test_set:
                neutral, cation, anion = triplet
                _, m = total_loss(
                    params, compute_energy_fn, neutral, cation, anion, 
                    ground_truth_energy=neutral.energy, janak_weight=JANAK_WEIGHT
                )
                test_losses.append(m['total_loss'])
            print(f"  >>> Validation Loss: {jnp.mean(test_losses):.5f}")

        if epoch % 20 == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch}_params.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(jax.device_get(params), f)
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    final_path = os.path.join(CKPT_DIR, "final_model.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    print(f"\nTraining Complete. Model saved to {final_path}")
    
    plot_loss_history(loss_history)

if __name__ == "__main__":
    train()