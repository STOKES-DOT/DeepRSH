import os; os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import jax.tree_util as jtu


def elu(x):
    return jax.nn.elu(x)


def leaky_relu(x, negative_slope=0.2):
    return jax.nn.leaky_relu(x, negative_slope)


def layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


# 为 Linear 注册 PyTree 接口
@jtu.register_pytree_node_class
@dataclass
class Linear:
    """
    - x: (..., in_features)
    - W: (in_features, out_features)
    - b: (out_features,)
    """
    W: jnp.ndarray
    b: Optional[jnp.ndarray] = None

    @staticmethod
    def init(key, in_features, out_features, bias=True, scale=1.0):
        # Xavier uniform
        lim = jnp.sqrt(6.0 / (in_features + out_features)) * scale
        kW, kb = random.split(key)
        W = random.uniform(kW, (in_features, out_features), minval=-lim, maxval=lim)
        b_arr = random.uniform(kb, (out_features,), minval=-lim, maxval=lim) if bias else None
        if bias:
            pass
        return Linear(W=W, b=b_arr)

    def __call__(self, x):
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y
    
    def tree_flatten(self):
        # 将 Linear 对象展平为子节点和辅助数据
        if self.b is not None:
            children = (self.W, self.b)
        else:
            children = (self.W,)
        aux_data = (self.b is not None,)  # 存储是否有偏置的信息
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        has_bias = aux_data[0]
        if has_bias:
            W, b = children
            return cls(W=W, b=b)
        else:
            W = children[0]
            return cls(W=W, b=None)


def dropout(rng, x, rate: float, deterministic: bool):
    if deterministic or rate == 0.0:
        return x
    keep_prob = 1.0 - rate
    mask = random.bernoulli(rng, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0.0)


# 为 GATLayerBaseParams 注册 PyTree 接口
@jtu.register_pytree_node_class
@dataclass
class GATLayerBaseParams:
    head_projections: Tuple[Linear, ...]               # len = num_heads
    scoring_fn_target: jnp.ndarray                     # (num_heads, out_features, 1)
    scoring_fn_source: jnp.ndarray                     # (num_heads, out_features, 1)
    edge_distance_proj: Linear                         # in=1 out=num_heads, bias=False
    edge_bond_proj: Linear                             # in=1 out=num_heads, bias=False
    skip_proj: Optional[Linear]                        # in=in_features out=num_heads*out_features, bias=False
    bias: Optional[jnp.ndarray]                        # (num_heads*out_features,) if concat else (out_features,)
    
    def tree_flatten(self):
        # 将所有属性作为子节点
        children = (
            self.head_projections,
            self.scoring_fn_target,
            self.scoring_fn_source,
            self.edge_distance_proj,
            self.edge_bond_proj,
            self.skip_proj,
            self.bias,
        )
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            head_projections,
            scoring_fn_target,
            scoring_fn_source,
            edge_distance_proj,
            edge_bond_proj,
            skip_proj,
            bias,
        ) = children
        return cls(
            head_projections=head_projections,
            scoring_fn_target=scoring_fn_target,
            scoring_fn_source=scoring_fn_source,
            edge_distance_proj=edge_distance_proj,
            edge_bond_proj=edge_bond_proj,
            skip_proj=skip_proj,
            bias=bias,
        )


# 注意：GATLayerConfig 不需要 PyTree 注册，因为它只包含配置参数，不包含可训练参数
@dataclass
class GATLayerConfig:
    num_in_features: int
    num_out_features: int
    num_heads: int = 1
    concat: bool = False
    add_skip_connection: bool = True
    bias: bool = True
    dropout_prob: float = 0.6
    activation: Optional[Callable] = elu
    log_attention_weights: bool = True
    esp: float = 1e-6


def init_gat_layer_params(key, cfg: GATLayerConfig) -> GATLayerBaseParams:
    k = key
    keys = random.split(k, 3 + cfg.num_heads + (1 if cfg.add_skip_connection else 0) + (1 if cfg.bias else 0))

    # head projections (bias=False)
    head_proj_keys = keys[:cfg.num_heads]
    head_projections = tuple(
        Linear.init(hk, cfg.num_in_features, cfg.num_out_features, bias=False)
        for hk in head_proj_keys
    )

    idx = cfg.num_heads

    # scoring tensors: Xavier-ish init
    k_t = keys[idx]; idx += 1
    k_s = keys[idx]; idx += 1
    lim = jnp.sqrt(6.0 / (cfg.num_out_features + 1.0))
    scoring_fn_target = random.uniform(k_t, (cfg.num_heads, cfg.num_out_features, 1), minval=-lim, maxval=lim)
    scoring_fn_source = random.uniform(k_s, (cfg.num_heads, cfg.num_out_features, 1), minval=-lim, maxval=lim)

    # edge proj: Linear(1 -> num_heads, bias=False)
    k_ed = keys[idx]; idx += 1
    k_eb = keys[idx]; idx += 1
    edge_distance_proj = Linear.init(k_ed, 1, cfg.num_heads, bias=False)
    edge_bond_proj = Linear.init(k_eb, 1, cfg.num_heads, bias=False)

    # bias
    b_arr = None
    if cfg.bias:
        kb = keys[idx]; idx += 1
        if cfg.concat:
            b_arr = jnp.zeros((cfg.num_heads * cfg.num_out_features,), dtype=jnp.float32)
        else:
            b_arr = jnp.zeros((cfg.num_out_features,), dtype=jnp.float32)

    # skip proj
    skip_proj = None
    if cfg.add_skip_connection:
        ks = keys[idx]; idx += 1
        skip_proj = Linear.init(ks, cfg.num_in_features, cfg.num_heads * cfg.num_out_features, bias=False)

    return GATLayerBaseParams(
        head_projections=head_projections,
        scoring_fn_target=scoring_fn_target,
        scoring_fn_source=scoring_fn_source,
        edge_distance_proj=edge_distance_proj,
        edge_bond_proj=edge_bond_proj,
        skip_proj=skip_proj,
        bias=b_arr,
    )


def skip_concat_bias(
    cfg: GATLayerConfig,
    params: GATLayerBaseParams,
    attention_coefficients: jnp.ndarray,   # (H, N, N)
    in_nodes_features: jnp.ndarray,        # (N, Fin)
    out_nodes_features: jnp.ndarray,       # (H, N, Fout) or reshaped equivalent
):

    if out_nodes_features.ndim != 3 or out_nodes_features.shape[0] != cfg.num_heads:
        out_nodes_features = out_nodes_features.reshape(cfg.num_heads, -1, cfg.num_out_features)

    # skip connection
    if cfg.add_skip_connection:
        if cfg.num_out_features == cfg.num_in_features:
            out_nodes_features = out_nodes_features + in_nodes_features[None, :, :]
        else:
            assert params.skip_proj is not None
            skip = params.skip_proj(in_nodes_features)  # (N, H*Fout)
            skip = skip.reshape(-1, cfg.num_heads, cfg.num_out_features).transpose(1, 0, 2)  # (H, N, Fout)
            out_nodes_features = out_nodes_features + skip

    # concat or average heads
    if cfg.concat:
        out = out_nodes_features.transpose(1, 0, 2).reshape(-1, cfg.num_heads * cfg.num_out_features)  # (N, H*Fout)
    else:
        out = jnp.mean(out_nodes_features, axis=0)  # (N, Fout)

    # bias
    if params.bias is not None:
        out = out + params.bias

    # activation
    if cfg.activation is None:
        return out
    return cfg.activation(out)


def gat_forward(
    rng,
    cfg: GATLayerConfig,
    params: GATLayerBaseParams,
    nodes_features: jnp.ndarray,              # (N, Fin)
    degree_matrix: jnp.ndarray,               # (N, N)
    edges_features_distance: jnp.ndarray,     # (N, N) or (N, N, 1)  
    edges_features_bond: jnp.ndarray,         # (N, N) or (N, N, 1)                
    deterministic: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:

    N = nodes_features.shape[0]
    connectivity_mask = jnp.where(degree_matrix > 0, 0.0, -1e9)
    assert connectivity_mask.shape == (N, N)
    head_feats = []
    for h in range(cfg.num_heads):
        hf = params.head_projections[h](nodes_features)
        head_feats.append(hf)
    nodes_features_proj = jnp.stack(head_feats, axis=0)

    # dropout on projected node features
    rng, kdrop = random.split(rng)
    nodes_features_proj = dropout(kdrop, nodes_features_proj, rate=cfg.dropout_prob, deterministic=deterministic)

    # scores_source/target: bmm => (H, N, 1)
    scores_source = jnp.matmul(nodes_features_proj, params.scoring_fn_source)  # (H, N, 1)
    scores_target = jnp.matmul(nodes_features_proj, params.scoring_fn_target)  # (H, N, 1)

    # all_scores: (H, N, N)
    all_scores = leaky_relu(scores_source + jnp.swapaxes(scores_target, 1, 2), 0.2)
    ed = edges_features_distance
    if ed.ndim == 3 and ed.shape[-1] == 1:
        ed = ed[..., 0]
    ed_in = (-ed).reshape(-1, 1)  # (N*N,1)
    ed_out = params.edge_distance_proj(ed_in)  # (N*N,H)
    edge_distance_contribution = ed_out.reshape(N, N, cfg.num_heads).transpose(2, 0, 1)  # (H,N,N)
    all_scores = all_scores + edge_distance_contribution

    # edge bond contribution
    eb = edges_features_bond
    if eb.ndim == 3 and eb.shape[-1] == 1:
        eb = eb[..., 0]
    eb_in = eb.reshape(-1, 1)  # (N*N,1)
    eb_out = params.edge_bond_proj(eb_in)  # (N*N,H)
    edge_bond_contribution = eb_out.reshape(N, N, cfg.num_heads).transpose(2, 0, 1)  # (H,N,N)
    all_scores = all_scores + edge_bond_contribution
    masked_scores = all_scores + connectivity_mask[None, :, :]
    max_vals = jnp.max(masked_scores, axis=-1, keepdims=True)
    stable_scores = masked_scores - max_vals
    all_attention_coefficients = jax.nn.softmax(stable_scores, axis=-1)  # (H,N,N)
    out_nodes_features = jnp.matmul(all_attention_coefficients, nodes_features_proj)  # (H,N,Fout)

    updated_nodes_features = skip_concat_bias(
        cfg, params, all_attention_coefficients, nodes_features, out_nodes_features
    )
    unf = jax.lax.stop_gradient(updated_nodes_features)

    node_similarity = jnp.matmul(unf, unf.T)  # (N,N)
    distance_decay = -ed  # (N,N)

    updated_connectivity_mask = jax.nn.sigmoid(node_similarity) * distance_decay * (degree_matrix > 0).astype(jnp.float32)

   
    updated_connectivity_mask = updated_connectivity_mask + cfg.esp
    updated_connectivity_mask = layer_norm(updated_connectivity_mask, eps=1e-5)
    updated_connectivity_mask = updated_connectivity_mask + updated_connectivity_mask.T

    attention_weights = all_attention_coefficients if cfg.log_attention_weights else None
    return updated_nodes_features, updated_connectivity_mask, attention_weights