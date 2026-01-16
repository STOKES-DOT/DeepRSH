from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import os; os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import jax
import jax.numpy as jnp
from jax import random
import jax.tree_util as jtu

def elu(x):
    return jax.nn.elu(x)

@jax.tree_util.register_pytree_node_class
@dataclass
class Linear:
    W: jnp.ndarray
    b: jnp.ndarray

    @staticmethod
    def init(key, in_features: int, out_features: int) -> "Linear":
        lim = jnp.sqrt(6.0 / (in_features + out_features))
        kW, kb = random.split(key)
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

@jax.tree_util.register_pytree_node_class
@dataclass
class MLP:
    layers: Tuple[Linear, ...]
    activation: Callable = elu

    @staticmethod
    def init(key, sizes: List[int], activation: Callable = elu) -> "MLP":
        keys = random.split(key, len(sizes) - 1)
        layers = tuple(Linear.init(k, sizes[i], sizes[i + 1]) for i, k in enumerate(keys))
        return MLP(layers=layers, activation=activation)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def tree_flatten(self):
        children = self.layers 
        aux_data = self.activation 
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(layers=children, activation=aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass
class NodesEmbeddingParams:
    mlp: MLP

    def tree_flatten(self):
        children = (self.mlp,)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (mlp,) = children
        return cls(mlp=mlp)


def init_nodes_embedding_params(key: jax.Array, nodes_size: int) -> NodesEmbeddingParams:
    sizes = [nodes_size, nodes_size**2, nodes_size**2, nodes_size**2, nodes_size]
    mlp = MLP.init(key, sizes=sizes, activation=elu)
    return NodesEmbeddingParams(mlp=mlp)


@dataclass
class NodesEmbedding:
    atom_embedding_module: object
    ref_mol2: str

    def __post_init__(self):
        atom_embed = self.atom_embedding_module.AtomEmbedding(self.ref_mol2)
        atom_type_vector, atom_part, atom_charge = atom_embed()
        self.nodes_size = int(jnp.asarray(atom_type_vector).shape[1])
        self.ref_atom_charge = jnp.asarray(atom_charge, dtype=jnp.float32)

    def forward(
        self,
        params: NodesEmbeddingParams,
        mol2: str,
    ) -> jnp.ndarray:
        atom_embed = self.atom_embedding_module.AtomEmbedding(mol2)
        atom_type_vector, atom_part, atom_charge = atom_embed()

        atom_type_vector = jnp.asarray(atom_type_vector, dtype=jnp.float32)
        atom_part = jnp.asarray(atom_part, dtype=jnp.int32)

        part_vec = jnp.repeat(atom_part[:, None].astype(jnp.float32), self.nodes_size, axis=1)
        charge_per_atom = self.ref_atom_charge[atom_part]
        charge_vec = jnp.repeat(charge_per_atom[:, None], self.nodes_size, axis=1)

        x = atom_type_vector + part_vec + charge_vec

        def embed_one(v):
            return params.mlp(v)

        return jax.vmap(embed_one)(x)


if __name__ == "__main__":
    import atom_embedding

    key = random.PRNGKey(0)
    mol2 = "/home/yjiao/DeepRSH/module/net.mol2"

    model = NodesEmbedding(atom_embedding_module=atom_embedding, ref_mol2="/home/yjiao/DeepRSH/module/net.mol2")
    params = init_nodes_embedding_params(key, model.nodes_size)

    out = model.forward(params, mol2)
    print(out)