import numpy as onp
import random as pyrandom
import igraph as ig
from einops import rearrange
from functools import partial
import jax
import jax.numpy as jnp
from jax import random

from functions.synthetic.abstract import GraphModel
from functions.synthetic.graph_utils import *


class ErdosRenyi(GraphModel):
    """
    Erdos-Renyi random graph
    
    Args:
        edges_per_var (float): expected number of edges per node, scaled to account for number of nodes
    """
    def __init__(self, edges_per_var):
        self.edges_per_var = edges_per_var

    def __call__(self, rng, n_vars):
        # select p s.t. we get requested edges_per_var in expectation
        n_edges = self.edges_per_var * n_vars
        p = min(n_edges / ((n_vars * (n_vars - 1)) / 2), 0.99)

        # sample
        mat = rng.binomial(n=1, p=p, size=(n_vars, n_vars)).astype(int) # bernoulli

        # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
        dag = onp.tril(mat, k=-1)

        # randomly permute
        p = rng.permutation(onp.eye(n_vars).astype(int))
        dag = p.T @ dag @ p
        return dag


class ScaleFree(GraphModel):
    """
    Barabasi-Albert (scale-free)
    Power-law in-degree
    
    Args:
        edges_per_var (int): number of edges per node
        power (float): power in preferential attachment process. 
            Higher values make few nodes have high in-degree.

    """
    def __init__(self, edges_per_var, power=1.0):
        self.edges_per_var = edges_per_var
        self.power = power

    def __call__(self, rng, n_vars):
        pyrandom.seed(rng.bit_generator.state["state"]["state"]) # seed pyrandom based on state of numpy rng
        _ = rng.normal() # advance rng state by 1
        perm = rng.permutation(n_vars).tolist()
        g = ig.Graph.Barabasi(n=n_vars, m=self.edges_per_var, directed=True, power=self.power).permute_vertices(perm)
        mat = graph_to_mat(g)
        return mat

    
# class ErdosRenyi_vectorized():
#     """
#     Erdos-Renyi random graph, 
    
#     Args:
#         edges_per_var (float): expected number of edges per node, scaled to account for number of nodes
#     """
#     def __init__(self, edges_per_var):
#         self.edges_per_var = edges_per_var

#     def __call__(self, rng, n_vars, shape):
        
#         # for i, dim in enumerate(shape):
#         #     globals()[f'dim_{i}'] = dim
        
#         n = onp.prod(shape)
        
#         # select p s.t. we get requested edges_per_var in expectation
#         n_edges = self.edges_per_var * n_vars
#         p = min(n_edges / ((n_vars * (n_vars - 1)) / 2), 0.99)

#         # sample
#         mat = rng.binomial(n=1, p=p, size=(n_vars, n_vars)).astype(int) # bernoulli

#         # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
#         dag = onp.tril(mat, k=-1)

#         #-------
#         a1 = (onp.eye(n_vars).astype(int))
#         array = onp.repeat(a1[onp.newaxis, :, :], n, axis=-3)

#         P = onp.array([rng.permutation(subarray, axis=-2) for subarray in array])
#         # print(f'P: {P.shape}')

#         dag_final = P.swapaxes(-2,-1) @ dag @ P

#         shape_final = shape + [n_vars, n_vars]
#         dag_final = dag_final.reshape(shape_final)
        
#         #-------
        
#         return dag_final




class ErdosRenyi_jax(GraphModel):
    """
    Erdos-Renyi random graph, JAX implementation
    
    Args:
        edges_per_var (float): expected number of edges per node, scaled to account for number of nodes
    """
    def __init__(self, edges_per_var):
        self.edges_per_var = edges_per_var

    def __call__(self, rng, n_vars):
        # select p s.t. we get requested edges_per_var in expectation
        key_b, key_p = jax.random.split(rng, 2)
        
        n_edges = self.edges_per_var * n_vars
        p = min(n_edges / ((n_vars * (n_vars - 1)) / 2), 0.99)

        # sample
        mat = jax.random.bernoulli(key_b, p=p, shape=(n_vars, n_vars))

        # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
        dag = jnp.tril(mat, k=-1)

        # randomly permute
        p = jax.random.permutation(key_p, jnp.eye(n_vars).astype(int))

        dag = p.T @ dag @ p
        return dag


class ScaleFree_jax(GraphModel):
    """
    Barabasi-Albert (scale-free)
    Power-law in-degree
    
    Args:
        edges_per_var (int): number of edges per node
        power (float): power in preferential attachment process. 
            Higher values make few nodes have high in-degree.

    """
    def __init__(self, edges_per_var):
        self.edges_per_var = edges_per_var

    def __call__(self, rng, n_vars):
        
        key1, key2 = jax.random.split(rng, 2)        
        
        # get undirected Barabasi scale-free graph adjacency amtrix
        mat = get_Barabasi_graph(key1, n_vars, self.edges_per_var, initial_graph=None)
        
        dag = jnp.tril(mat, k=-1)

        # randomly permute
        p = jax.random.permutation(key2, jnp.eye(n_vars).astype(int))

        dag = p.T @ dag @ p

        return dag
    

def _random_subset(seq: jnp.ndarray, m: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
    targets = jnp.zeros((m))
    i=0
    while i < m:
        rng_key, subkey = random.split(rng_key)
        x = random.choice(subkey, seq)
        targets = targets.at[i].set(x)
        i+=1
    return jnp.array(list(targets), dtype=int)


@partial(jax.jit, static_argnames=['edges_per_var', 'n_vars'])
def get_Barabasi_graph(key, n_vars, edges_per_var, initial_graph=None):
        
        n = n_vars
        m = edges_per_var
        
        if m < 1 or m >= n:
            raise ValueError(
                f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
            )

        rng_key = key

        if initial_graph is None:
            G = jnp.zeros((m + 1, m + 1))
            G = G.at[:m, m].set(1)
            G = G.at[m, :m].set(1)
        else:
            G = initial_graph
            
        # List of existing nodes, with nodes repeated once for each adjacent edge
        # repeated_nodes = jnp.array([n for n, d in enumerate(jnp.sum(G.astype(jnp.int32), axis=0)) for _ in range(d)]) #------------- changed here
        repeated_nodes = jnp.concatenate([jnp.arange((m+1)-1),jnp.repeat((m+1)-1, (m+1)-1)])

        for source in range(G.shape[0], n):
            
            rng_key, subkey = random.split(rng_key)
            targets = _random_subset(repeated_nodes, m, subkey)

            new_row = jnp.zeros((1, source))
            new_row = new_row.at[:, targets].set(1)
            new_col = jnp.zeros((source + 1, 1))
            new_col = new_col.at[targets, :].set(1)

            G = jnp.block([[G], [new_row]])
            G = jnp.block([G, new_col])

            repeated_nodes = jnp.concatenate([repeated_nodes, targets, jnp.repeat(source, m)])

        return G.astype(jnp.int32)



