import numpy as np
from scipy.optimize import minimize
from energy_get import EnergyGetter
import jax.numpy as jnp
import jax
class OPTXC:
    def __init__(self, mol2):
        self.mol2 = mol2
        self.initial_omega = 0.3
        self.initial_alpha = 0.8
        self.initial_beta = 0.2
    def get_loss(self, params):
        omega, alpha, beta = params
        energy_getter = EnergyGetter(
            self.mol2, 
            alpha=alpha, 
            beta=beta, 
            omega=omega
        )
        j, _, _, _ = energy_getter.forward()
        loss = j
        return loss

    def numerical_grad(self, params, eps=1e-5):

        omega, alpha, beta = params
        loss0 = self.get_loss(params)
        
        # 对omega的梯度
        grad_omega = (self.get_loss((omega + eps, alpha, beta)) - loss0) / eps
        
        # 对alpha的梯度
        grad_alpha = (self.get_loss((omega, alpha + eps, beta)) - loss0) / eps
        
        # 对beta的梯度
        grad_beta = (self.get_loss((omega, alpha, beta + eps)) - loss0) / eps
        
        return jnp.array([grad_omega, grad_alpha, grad_beta])
    
    def optimize(self, steps=200, lr=0.001):
        """
        简单的梯度下降优化
        """
        # 初始化参数
        params = (
            jnp.array(float(self.initial_omega)),
            jnp.array(float(self.initial_alpha)), 
            jnp.array(float(self.initial_beta))
        )
        print(params)
        print(f"初始参数: omega={params[0]:.6f}, alpha={params[1]:.6f}, beta={params[2]:.6f}")
        
        for step in range(steps):
            # 计算梯度
            grads = self.numerical_grad(params)
            
            # 更新参数
            new_omega = float(f"{(params[0] - lr * grads[0]):.10f}")
            new_alpha = float(f"{(params[1] - lr * grads[1]):.10f}")
            new_beta = float(f"{(params[2] - lr * grads[2]):.10f}")
            
            params = np.array([abs(new_omega), abs(new_alpha), abs(new_beta)])
            # 计算当前损失
            loss = self.get_loss(params)
            
            print(f"步骤 {step+1}: 损失={loss:.6f}, "
                  f"omega={params[0]:.6f}, alpha={params[1]:.6f}, beta={params[2]:.6f}")
        
        return params
    
if __name__ == '__main__':
    mol2 = '/home/yjiao/DeepRSH/module/net.mol2'
    optxc = OPTXC(mol2)
    
    initial_params = (
        jnp.array(optxc.initial_omega),
        jnp.array(optxc.initial_alpha),
        jnp.array(optxc.initial_beta)
    )
    loss = optxc.get_loss(initial_params)
    print('初始损失:', loss)

    print("\n开始优化:")
    final_params = optxc.optimize(steps=100, lr=0.5)