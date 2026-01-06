import os
import pickle

from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split
from optax import adam, apply_updates

from grad_dft import energy_predictor, Molecule, loader

from loss import total_loss
from xc_functional import build_functional

# ==========================================
# 数据加载与绘图
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
# 主训练函数
# ==========================================

def train():
    # 参数配置
    SEED = 42
    LR = 1e-4
    MOMENTUM = 0.9
    EPOCHS = 100
    JANAK_WEIGHT = 1.0
    DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "h2_dataset.hdf5")
    CKPT_DIR = "ckpts/RSH_Janak"
    
    # 初始化网络
    key = PRNGKey(SEED)
    functional = build_functional()
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
            janak_weight=JANAK_WEIGHT,
        )
        (cost, metrics), grads = loss_fn(params, neutral, cation, anion)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = apply_updates(params, updates)
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
            epoch_metrics['mse'].append(m['mean_sq_error'])
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
                eval_loss = partial(
                    total_loss,
                    compute_energy=compute_energy_fn,
                    ground_truth_energy=neutral.energy,
                    janak_weight=JANAK_WEIGHT,
                )
                (eval_cost, eval_metrics), _ = eval_loss(params, neutral, cation, anion)
                test_losses.append(eval_cost)
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
