import igraph as ig
import numpy as onp
import jax.numpy as jnp
import functools 

import igraph as ig
import jax.numpy as jnp
from jax import jit, vmap


def graph_to_mat(g):
    """Returns adjacency matrix of ig.Graph object """
    return onp.array(g.get_adjacency().data).astype(int)

def mat_to_graph(mat):
    """Returns ig.Graph object for adjacency matrix """
    return ig.Graph.Weighted_Adjacency(mat.tolist())

def graph_to_toporder(g):
    """Returns adjacency matrix of ig.Graph object """
    return onp.array(g.topological_sorting()).astype(int)

def mat_to_toporder(mat):
    """Returns adjacency matrix of ig.Graph object """
    return onp.array(mat_to_graph(mat).topological_sorting()).astype(int)

def graph_to_mat_jax(g):
    """Returns adjacency matrix of ``ig.Graph`` object

    Args:
        g (igraph.Graph): graph

    Returns:
        ndarray:
        adjacency matrix

    """
    return jnp.array(g.get_adjacency().data)

@functools.partial(jit, static_argnums=(1,))
def acyclic_constr_nograd(mat, n_vars):
    """
    Differentiable acyclicity constraint from Yu et al. (2019)
    http://proceedings.mlr.press/v97/yu19a/yu19a.pdf

    Args:
        mat (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
        n_vars (int): number of variables, to allow for ``jax.jit``-compilation

    Returns:
        constraint value ``[1, ]``
    """

    alpha = 1.0 / n_vars
    # M = jnp.eye(n_vars) + alpha * mat * mat # [original version]
    M = jnp.eye(n_vars) + alpha * mat

    M_mult = jnp.linalg.matrix_power(M, n_vars)
    h = jnp.trace(M_mult) - n_vars
    return h

elwise_acyclic_constr_nograd = jit(vmap(acyclic_constr_nograd, (0, None), 0), static_argnums=(1,))
