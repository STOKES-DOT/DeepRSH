import jax.numpy as jnp
from jax.nn import gelu
from jaxtyping import Array, Float

import grad_dft as gd
from grad_dft import (
    NeuralFunctional,
    Molecule,
    canonicalize_inputs,
    correlation_polarization_correction,
    exchange_polarization_correction,
)


def lsda_x_e(rho: Float[Array, "grid spin"], clip_cte) -> Float[Array, "grid"]:
    rho = jnp.clip(rho, a_min=clip_cte)
    lda_es = (
        -3.0
        / 4.0
        * (jnp.array([[3.0, 6.0]]) / jnp.pi) ** (1 / 3)
        * (rho.sum(axis=1, keepdims=True)) ** (4 / 3)
    )
    return exchange_polarization_correction(lda_es, rho)


def b88_x_e(
    rho: Float[Array, "grid spin"],
    grad_rho: Float[Array, "grid spin dimension"],
    clip_cte: float = 1e-30,
) -> Float[Array, "grid"]:
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

    e_PF = A / 2 * (
        2 * jnp.log(x)
        - jnp.log(X)
        + 2 * b / Q * jnp.arctan(Q / (2 * x + b))
        - b
        * x0
        / X0
        * (
            jnp.log((x - x0) ** 2 / X)
            + 2 * (2 * x0 + b) / Q * jnp.arctan(Q / (2 * x + b))
        )
    )
    e_tilde = correlation_polarization_correction(e_PF, rho, clip_cte)
    return e_tilde * rho.sum(axis=1)


def lyp_c_e(
    rho: Float[Array, "grid spin"],
    grad_rho: Float[Array, "grid spin 3"],
    grad2rho: Float[Array, "grid spin"],
    clip_cte=1e-30,
) -> Float[Array, "grid"]:
    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    CF = (3 / 10) * (3 * jnp.pi**2) ** (2 / 3)

    rho = jnp.clip(rho, a_min=clip_cte)
    grad_rho_norm_sq = jnp.sum(grad_rho**2, axis=-1)
    t = (jnp.where(rho > clip_cte, grad_rho_norm_sq / rho, 0) - grad2rho) / 8.0
    frac = jnp.where(
        rho.sum(axis=1) > clip_cte, ((rho**2).sum(axis=1)) / (rho.sum(axis=1)) ** 2, 1
    )
    gamma = 2 * (1 - frac)
    rhos_ts = rho.sum(axis=1) * t.sum(axis=1)
    rho_t = (rho * t).sum(axis=1)
    rho_grad2rho = (rho * grad2rho).sum(axis=1)
    rhom1_3 = (rho.sum(axis=1)) ** (-1 / 3)
    rho8_3 = (rho ** (8 / 3)).sum(axis=1)
    rhom5_3 = (rho.sum(axis=1)) ** (-5 / 3)
    exp_factor = jnp.where(rho.sum(axis=1) > 0, jnp.exp(-c * rhom1_3), 0)
    parenthesis = 2 ** (2 / 3) * CF * (rho8_3) - rhos_ts + rho_t / 9 + rho_grad2rho / 18
    braket_m_rho = jnp.where(
        rho.sum(axis=1) > clip_cte, 2 * b * rhom5_3 * parenthesis * exp_factor, 0.0
    )

    return -a * jnp.where(
        rho.sum(axis=1) > clip_cte,
        gamma / (1 + d * rhom1_3) * (rho.sum(axis=1) + braket_m_rho),
        0.0,
    )


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


def build_functional():
    return NeuralFunctional(
        coefficients=nn_coefficients,
        energy_densities=energy_density_rsh,
        coefficient_inputs=gd.dm21_coefficient_inputs,
    )


__all__ = [
    "build_functional",
    "energy_density_rsh",
    "nn_coefficients",
    "lsda_x_e",
    "b88_x_e",
    "vwn_c_e",
    "lyp_c_e",
]
