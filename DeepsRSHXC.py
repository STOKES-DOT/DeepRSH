from __future__ import annotations
import os; os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
from dataclasses import dataclass
from decimal import Decimal
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import register_pytree_node_class
from nodes_embedding import NodesEmbedding, NodesEmbeddingParams, init_nodes_embedding_params
from bond_embedding import BondEmbedding, GaussianBasisParams, init_gaussian_basis_params
from GAT_Layer import GATLayerConfig, init_gat_layer_params, gat_forward


def _dropout(rng, x, rate: float, deterministic: bool):
    if deterministic or rate == 0.0:
        return x
    keep_prob = 1.0 - rate
    mask = random.bernoulli(rng, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0.0)

@register_pytree_node_class
@dataclass
class Linear:
    W: jnp.ndarray
    b: jnp.ndarray

    @staticmethod
    def init(key, in_features: int, out_features: int) -> "Linear":
        lim = jnp.sqrt(6.0 / (in_features + out_features))
        kW, _ = random.split(key)
        W = random.uniform(kW, (in_features, out_features), minval=-lim, maxval=lim).astype(jnp.float32)
        b = jnp.zeros((out_features,), dtype=jnp.float32)
        return Linear(W=W, b=b)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.W + self.b
    
    def tree_flatten(self):
        children = (self.W, self.b)  
        aux_data = None  
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        W, b = children
        return cls(W=W, b=b)


def _global_gated_attention_pool(x: jnp.ndarray, gate: Linear, proj: Linear) -> jnp.ndarray:
    g = jax.nn.sigmoid(gate(x))
    num = jnp.sum(g * x, axis=0)
    den = jnp.sum(g, axis=0) + 1e-12
    pooled = num / den
    return proj(pooled)

@register_pytree_node_class
@dataclass
class ParameterNN:
    l1: Linear
    l2: Linear
    l3: Linear
    l4: Linear
    l5: Linear
    l6: Linear
    l7: Linear
    l8: Linear

    @staticmethod
    def init(key, in_dim: int, hidden: int = 128, out_dim: int = 3) -> "ParameterNN":
        ks = random.split(key, 8)
        return ParameterNN(
            l1=Linear.init(ks[0], in_dim, hidden),
            l2=Linear.init(ks[1], hidden, hidden),
            l3=Linear.init(ks[2], hidden, hidden),
            l4=Linear.init(ks[3], hidden, hidden),
            l5=Linear.init(ks[4], hidden, hidden),
            l6=Linear.init(ks[5], hidden, hidden),
            l7=Linear.init(ks[6], hidden, hidden),
            l8=Linear.init(ks[7], hidden, out_dim),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.elu(self.l1(x))
        x = jax.nn.elu(self.l2(x))
        x = jax.nn.elu(self.l3(x))
        x = jax.nn.elu(self.l4(x))
        x = jax.nn.elu(self.l5(x))
        x = jax.nn.elu(self.l6(x))
        x = jax.nn.elu(self.l7(x))
        x = jax.nn.sigmoid(self.l8(x))
        return x
    
    def tree_flatten(self):
        children = (self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8)
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        l1, l2, l3, l4, l5, l6, l7, l8 = children
        return cls(l1=l1, l2=l2, l3=l3, l4=l4, l5=l5, l6=l6, l7=l7, l8=l8)

@register_pytree_node_class
@dataclass
class DeepsRSHXCParams:
    nodes: NodesEmbeddingParams
    gb: GaussianBasisParams
    gat_layers: Tuple[object, ...]
    pool_gate: Linear
    pool_proj: Linear
    param_nn: ParameterNN
    
    def tree_flatten(self):
        children = (self.nodes, self.gb, self.gat_layers, self.pool_gate, self.pool_proj, self.param_nn)
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nodes, gb, gat_layers, pool_gate, pool_proj, param_nn = children
        return cls(nodes=nodes, gb=gb, gat_layers=gat_layers, 
                  pool_gate=pool_gate, pool_proj=pool_proj, param_nn=param_nn)


class DeepsRSHXC:
    def __init__(self, ref_mol2: str, num_heads: int = 8, num_gat_layers: int = 2, dropout: float = 0.6):
        self.ref_mol2 = ref_mol2
        self.num_heads = num_heads
        self.num_gat_layers = num_gat_layers
        self.dropout = dropout

        self.nodes_embed = NodesEmbedding(atom_embedding_module=__import__("atom_embedding"), ref_mol2=ref_mol2)
        self.nodes_size = self.nodes_embed.nodes_size

    def init(self, key) -> DeepsRSHXCParams:
        k1, k2, k3, k4, k5 = random.split(key, 5)

        nodes_params = init_nodes_embedding_params(k1, self.nodes_size)
        gb_params = init_gaussian_basis_params(init_sigma=1.0, init_mu=0.0)

        gat_cfg = GATLayerConfig(
            num_in_features=self.nodes_size,
            num_out_features=self.nodes_size,
            num_heads=self.num_heads,
            concat=False,
            add_skip_connection=True,
            dropout_prob=self.dropout,
            activation=jax.nn.elu,
            log_attention_weights=False,
            esp=1e-6,
        )
        gat_keys = random.split(k2, self.num_gat_layers)
        gat_layers = tuple(init_gat_layer_params(gk, gat_cfg) for gk in gat_keys)

        pool_gate = Linear.init(k3, self.nodes_size, self.nodes_size)
        pool_proj = Linear.init(k4, self.nodes_size, self.nodes_size)

        param_nn = ParameterNN.init(k5, self.nodes_size, hidden=128, out_dim=3)

        return DeepsRSHXCParams(
            nodes=nodes_params,
            gb=gb_params,
            gat_layers=gat_layers,
            pool_gate=pool_gate,
            pool_proj=pool_proj,
            param_nn=param_nn,
        )

    def forward(self, rng, params: DeepsRSHXCParams, mol2: str, deterministic: bool = True):
        node_feat = self.nodes_embed.forward(params.nodes, mol2)

        be = BondEmbedding(mol2)
        coords = jnp.asarray(be.atom_coords(), dtype=jnp.float32)
        edge_feat_dis = BondEmbedding.gaussian_basis_matrix_jax(coords, params.gb)
        edge_feat_bond = jnp.asarray(be.get_bond_type(), dtype=jnp.float32)
        degree = jnp.asarray(be.get_degree_matrix(), dtype=jnp.float32)
        _ = be.get_atom_pairs_direction()

        gat_cfg = GATLayerConfig(
            num_in_features=self.nodes_size,
            num_out_features=self.nodes_size,
            num_heads=self.num_heads,
            concat=False,
            add_skip_connection=True,
            dropout_prob=self.dropout,
            activation=jax.nn.elu,
            log_attention_weights=False,
            esp=1e-6,
        )

        x = node_feat
        connectivity_mask = None
        for i, layer_p in enumerate(params.gat_layers):
            rng, k1 = random.split(rng)
            x, connectivity_mask = gat_forward(k1, gat_cfg, layer_p, x, degree, edge_feat_dis, edge_feat_bond, deterministic=deterministic)[:2]
            if i < len(params.gat_layers) - 1:
                rng, k2 = random.split(rng)
                x = _dropout(k2, x, rate=self.dropout, deterministic=deterministic)

        graph_repr = _global_gated_attention_pool(x, params.pool_gate, params.pool_proj)
        p3 = params.param_nn(graph_repr)
        return p3
