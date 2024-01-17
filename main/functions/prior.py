# import distrax
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from einops import rearrange
from jax.scipy.stats import norm, multivariate_normal

from functions.synthetic.distributions import *
from functions.synthetic.graph import *


class Func_mu():
    def __init__(self, weights_distr, biases_distr, graph_distr):
        self.weights_distr = weights_distr
        self.biases_distr = biases_distr
        self.graph_distr = graph_distr

    def __call__(self, S, rng):
        
        d = S.shape[-1]
        
        W = self.weights_distr(rng, [d,d])
        b = self.biases_distr(rng, [1,d])
        G = self.graph_distr(rng, n_vars=d)
        
        # print(f'W: {W.shape}')
        # print(f'G: {G.shape}')
        # print(f'b: {b.shape}')
        
        # print(f'W: {W}')
        # print(f'G: {G}')
        # print(f'b: {b}')
        
        return S @ (G * W) + b

class LatentPrior():
    def __init__(self, weights_distr, biases_distr, graph_distr, noise_scale_distr, num_MC_samples):
        """
            weights_distr: distribution from which elements w_ij for weight matrix W [d d] are sampled
            biases_distr: distribution from which elements b_j for bias vector b [d] are sampled
            graph_distr: distribution from which graphs G [d d] are sampled
            graph: matrix G [d d] - if we're using ficed graph
            epsilon_std: standard deviation for natural noise distribution - now fixed, #TODO: later - distribution form which std is sampled
            num_MC_samples: number of MC samples used for MC approximation of log PDF p(S)
        """
        
        self.weights_distr = weights_distr
        self.biases_distr = biases_distr
        self.graph_distr = graph_distr
        
        self.noise_scale_distr = noise_scale_distr
        
        self.num_MC_samples = num_MC_samples

        
    def log_prob(self, S, key):
        """
           S = tensor [(...) N d] for example [s b N d] where s=num_s_samples, b=batch_size
        """
        # print(f'---prior')
        s = S.shape[-4]
        b = S.shape[-3]
        L = self.num_MC_samples
        
        key_mu, key_sig = jax.random.split(key)
        
        #---- getting mu
        keys = jax.random.split(key_mu, num=b*L)
        keys = rearrange(keys, '(b L) a -> b L a', b=b, L=L)
        
        func_mu = Func_mu(self.weights_distr, self.biases_distr, self.graph_distr)
        func_mu_MC = jax.vmap(func_mu, in_axes=(None, 0))                               # S fixed | keys 1:L to generate L parameter sets (W, G, b) for MC approximation of p(S)
        func_mu_MC_batch = jax.vmap(func_mu_MC, in_axes=(0, 0), out_axes=0)             # S 1:b   | keys 1:b to generate M parameter sets (W, G, b) for every dataset D1...Db in a batch for which this particular Sm is sampled
        func_mu_MC_batch_s = jax.vmap(func_mu_MC_batch, in_axes=(0, None), out_axes=0)  # S 1:M   | keys fixed to use the same parameter sets (W, G, b) for every S1...SM that were sampled given one dataset Db in a batch
        
        mu = func_mu_MC_batch_s(S, keys)
        
        # print(f'S: {S.shape}')
        # print(f'mu: {mu.shape}') # [s b L N d]
        # print(f'{mu[0][0]} - mu[0][0] of shape: {mu[0][0].shape} - for D1: S1')
        # print(f'{mu[1][0]} - mu[1][0] of shape: {mu[1][0].shape} - for D1: S2')
        # print(f'{mu[1][1]} - mu[1][0] of shape: {mu[1][1].shape} - for D2: S2')
        # print(f'mu: {mu}')
        
        #---- getting sig
        noise_std = self.noise_scale_distr(key_sig, shape=(mu.shape[1:]))
        noise_std = jnp.expand_dims(noise_std, axis=0)
        sig = jnp.repeat(noise_std, s, axis=0)
        
        print(f'{mu.shape} - mu in p(S) approximation [s b L N d]') # s=num_s_samples, b=batch_size, L=num_MC_samples
        print(f'{sig.shape} - sigma in p(S) approximation [s b L N d] where s=num_s_samples, b=batch_size, L=num_MC_samples')  # s=num_s_samples, b=batch_size, L=num_MC_samples
            
        S = jnp.expand_dims(S, axis=-3)
        
        log_probs = norm.logpdf(S, mu, sig)
        log_prob = log_probs.sum((-1,-2))   # [(...) L]
        
        # print(f'log_probs: {log_prob} - shape {log_prob.shape}')
        # print(f'log_probs: {log_probs.shape}')
        # print(f'log_prob: {log_prob.shape}')
        
        S_log_prob = logsumexp(log_prob, axis=-1) - jnp.array(self.num_MC_samples) # [(...)]

        # print(f'S_log_prob: {S_log_prob}')
        print('-------------------------')

        return S_log_prob
    