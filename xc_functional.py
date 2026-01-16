from functools import partial
from jax.random import split, PRNGKey
from jax import numpy as jnp, value_and_grad
from jax.nn import gelu, silu
import numpy as np
from optax import adam
from tqdm import tqdm
import os
from orbax.checkpoint import PyTreeCheckpointer

from grad_dft import (
    train_kernel, 
    energy_predictor,
    NeuralFunctional,
    canonicalize_inputs,
    Molecule,
    correlation_polarization_correction,
    exchange_polarization_correction,
)
import grad_dft as gd
from jaxtyping import Array, Float, PyTree, jaxtyped
from torch.utils.tensorboard import SummaryWriter
import jax
import json
def convert(o):
    if isinstance(o, np.float32):
        return float(o)  
    return o

squash_offset = 1e-4
layer_widths = [256] * 8
out_features = 20
sigmoid_scale_factor = 2.0
activation = silu

def nn_coefficients(instance, rhoinputs, *_, **__):
    x = gd.canonicalize_inputs(rhoinputs)  # Making sure dimensions are correct

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


def lsda_x_e(rho: Float[Array, "grid spin"], clip_cte) -> Float[Array, "grid"]:

    rho = jnp.clip(rho, a_min=clip_cte)
    lda_es = (
        -3.0
        / 4.0
        * (jnp.array([[3.0, 6.0]]) / jnp.pi) ** (1 / 3)
        * (rho.sum(axis=1, keepdims=True)) ** (4 / 3)
    )
    lda_e = exchange_polarization_correction(lda_es, rho)

    return lda_e

def b88_x_e(rho: Float[Array, "grid spin"], grad_rho: Float[Array, "grid spin dimension"], clip_cte: float = 1e-30) -> Float[Array, "grid"]:
    beta = 0.0042

    rho = jnp.clip(rho, a_min=clip_cte)

    # LDA preprocessing data: Note that we duplicate the density to sum and divide in the last eq.
    log_rho = jnp.log2(jnp.clip(rho, a_min=clip_cte))

    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)

    log_grad_rho_norm = jnp.log2(jnp.clip(grad_rho_norm_sq, a_min=clip_cte)) / 2

    # GGA preprocessing data
    log_x_sigma = log_grad_rho_norm - 4 / 3.0 * log_rho

    # assert not jnp.isnan(log_x_sigma).any() and not jnp.isinf(log_x_sigma).any()

    x_sigma = 2**log_x_sigma

    # Eq 2.78 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    b88_e = -(
        beta
        * 2
        ** (
            4 * log_rho / 3
            + 2 * log_x_sigma
            - jnp.log2(1 + 6 * beta * x_sigma * jnp.arcsinh(x_sigma))
        )
    ).sum(axis=1)

    # def fzeta(z): return ((1-z)**(4/3) + (1+z)**(4/3) - 2) / (2*(2**(1/3) - 1))
    # Eq 2.71 in from Time-Dependent Density-Functional Theory, from Carsten A. Ullrich
    # b88_e = b88_es[0] + (b88_es[1]-b88_es[0])*fzeta(zeta)

    return b88_e

def pw92_c_e(rho: Float[Array, "grid spin"], clip_cte: float = 1e-30) -> Float[Array, "grid"]:

    A_ = jnp.array([[0.031091, 0.015545]])
    alpha1 = jnp.array([[0.21370, 0.20548]])
    beta1 = jnp.array([[7.5957, 14.1189]])
    beta2 = jnp.array([[3.5876, 6.1977]])
    beta3 = jnp.array([[1.6382, 3.3662]])
    beta4 = jnp.array([[0.49294, 0.62517]])

    log_rho = jnp.log2(jnp.clip(rho.sum(axis=1, keepdims=True), a_min=clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_rho / 3.0
    brs_1_2 = 2 ** (log_rs / 2 + jnp.log2(beta1))
    ars = 2 ** (log_rs + jnp.log2(alpha1))
    brs = 2 ** (log_rs + jnp.log2(beta2))
    brs_3_2 = 2 ** (3 * log_rs / 2 + jnp.log2(beta3))
    brs2 = 2 ** (2 * log_rs + jnp.log2(beta4))

    e_PF = -2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2))

    e_tilde = correlation_polarization_correction(e_PF, rho, clip_cte)

    return e_tilde * rho.sum(axis = 1)

def vwn_c_e(rho: Float[Array, "grid spin"], clip_cte: float = 1e-30) -> Float[Array, "grid"]:
    A = jnp.array([[0.0621814, 0.0621814 / 2]])
    b = jnp.array([[3.72744, 7.06042]])
    c = jnp.array([[12.9352, 18.0578]])
    x0 = jnp.array([[-0.10498, -0.325]])

    rho = jnp.where(rho > clip_cte, rho, 0.0)
    log_rho = jnp.log2(jnp.clip(rho.sum(axis=1, keepdims=True), a_min=clip_cte))
    # assert not jnp.isnan(log_rho).any() and not jnp.isinf(log_rho).any()
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - log_rho / 3.0
    log_x = log_rs / 2
    rs = 2**log_rs
    x = 2**log_x

    X = 2 ** (2 * log_x) + 2 ** (log_x + jnp.log2(b)) + c
    X0 = x0**2 + b * x0 + c
    # assert not jnp.isnan(X).any() and not jnp.isinf(X0).any()

    Q = jnp.sqrt(4 * c - b**2)

    # check eq with https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/vwn.mpl
    e_PF = (
        A
        / 2
        * (
            2 * jnp.log(x)
            - jnp.log(X)
            + 2 * b / Q * jnp.arctan(Q / (2 * x + b))
            - b
            * x0
            / X0
            * (jnp.log((x - x0) ** 2 / X) + 2 * (2 * x0 + b) / Q * jnp.arctan(Q / (2 * x + b)))
        )
    )

    e_tilde = correlation_polarization_correction(e_PF, rho, clip_cte)

    # We have to integrate e = e_tilde * n as per eq 2.1 in original VWN article
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
    # assert not jnp.isnan(t).any() and not jnp.isinf(t).any()

    frac = jnp.where(
        rho.sum(axis=1) > clip_cte, ((rho**2).sum(axis=1)) / (rho.sum(axis=1)) ** 2, 1
    )
    gamma = 2 * (1 - frac)

    rhos_ts = rho.sum(axis=1) * t.sum(axis=1)
    # assert not jnp.isnan(rhos_ts).any() and not jnp.isinf(rhos_ts).any()

    rho_t = (rho * t).sum(axis=1)
    # assert not jnp.isnan(rho_t).any() and not jnp.isinf(rho_t).any()

    rho_grad2rho = (rho * grad2rho).sum(axis=1)
    # assert not jnp.isnan(rho_grad2rho).any() and not jnp.isinf(rho_grad2rho).any()

    rhom1_3 = (rho.sum(axis=1)) ** (-1 / 3)
    rho8_3 = (rho ** (8 / 3)).sum(axis=1)
    rhom5_3 = (rho.sum(axis=1)) ** (-5 / 3)

    exp_factor = jnp.where(rho.sum(axis=1) > 0, jnp.exp(-c * rhom1_3), 0)
    # assert not jnp.isnan(exp_factor).any() and not jnp.isinf(exp_factor).any()

    parenthesis = 2 ** (2 / 3) * CF * (rho8_3) - rhos_ts + rho_t / 9 + rho_grad2rho / 18

    braket_m_rho = jnp.where(rho.sum(axis=1) > clip_cte, 2 * b * rhom5_3 * parenthesis * exp_factor, 0.0)

    return -a * jnp.where(
        rho.sum(axis=1) > clip_cte, gamma / (1 + d * rhom1_3) * (rho.sum(axis=1) + braket_m_rho), 0.0
    )
    

def energy_density(molecule: gd.Molecule, clip_cte: float = 1e-30, *_, **__):
    rho = molecule.density()
    grad_rho = molecule.grad_density()
    grad2rho = molecule.lapl_density()
    
    b88_e = b88_x_e(rho, grad_rho, clip_cte)
    lda_e = lsda_x_e(rho, clip_cte)
    lyp_e = lyp_c_e(rho, grad_rho, grad2rho, clip_cte)
    vwn_e = vwn_c_e(rho, clip_cte)

    return jnp.stack((lda_e, b88_e, vwn_e, lyp_e), axis=1)

def combine_densities(densities, ehf):
    ehf = jnp.reshape(ehf, (ehf.shape[2], ehf.shape[0] * ehf.shape[1]))
    return jnp.concatenate((densities, ehf), axis=1)

def rsh_b3lyp_nn():
    return NeuralFunctional(
                            coefficients=nn_coefficients,
                            energy_densities=energy_density,
                            coefficient_inputs=gd.dm21_coefficient_inputs,
                            nograd_densities=lambda molecule, *_, **__: molecule.HF_energy_density(jnp.array([0.0, 0.3], dtype=jnp.float32)),
                            densitygrads=lambda self, params, molecule, nograd_densities, cinputs, grad_densities, *_, **__: gd.dm21_hfgrads_densities(
                                self, params, molecule, nograd_densities, cinputs, grad_densities, jnp.array([0.0, 0.3], dtype=jnp.float32)
                            ),
                            combine_densities=combine_densities,
                            nograd_coefficient_inputs=lambda molecule, *_, **__: molecule.HF_energy_density(jnp.array([0.0, 0.3], dtype=jnp.float32)),
                            coefficient_input_grads=lambda self, params, molecule, nograd_cinputs, grad_cinputs, densities, *_, **__: gd.dm21_hfgrads_cinputs(
                                self, params, molecule, nograd_cinputs, grad_cinputs, densities, jnp.array([0.0, 0.3], dtype=jnp.float32)
                                ),
                            combine_inputs=gd.dm21_combine_cinputs
                            )
