import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from einops import rearrange
from typing import Any, Sequence, Callable, NamedTuple, Optional, Tuple

from functions.transformer import *




def latent_space_samples(rng_z, mu, sigma, num_s_samples, num_sdec_samples):

    shape = (num_s_samples,) + sigma.shape
    epsilon = jax.random.normal(rng_z, shape)
    
    s = mu + epsilon[0:num_sdec_samples] * sigma
    return s
  


#----- VAE: sigma fixed

class Encoder2(nn.Module):
  """VAE Encoder"""
  
  num_layers: int
  num_layers_param1: int
  emb_dim: int
  num_heads: int
  ffn_dim_factor: int
  dropout_prob: float
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  @nn.compact
  def __call__(self, x, train):
    print(f'---VAE2: encoder') 
      
    #--------- separate
    # z = TransformerEncoder(self.num_layers, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k] 
    # x = TransformerEncoder(self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(z, train=train, fresh=False) # shape [b N d k]
    # mu = nn.Dense(1, name='enc_mean', kernel_init=self.kernel_init)(x) 
    # mu = rearrange(mu, 'b N d 1 -> b N d')
    # sigma = jnp.ones_like(mu) * 0.1
    
    #--------- same
    z = TransformerEncoder(self.num_layers, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k]
    mu = nn.Dense(1, name='enc_mean', kernel_init=self.kernel_init)(z) 
    mu = rearrange(mu, 'b N d 1 -> b N d')
    sigma = jnp.ones_like(mu) * 0.1
    
    
    
    #--------- Cov: avici-style
    B = nn.Dense(self.emb_dim, name='enc_cov', kernel_init=self.kernel_init)(z) # shape [b N d k]
    B = jnp.max(B, axis=-3) # shape [b d k]
    B_transp = jnp.swapaxes(B, -1, -2)
    C = B @ B_transp # shape [b d d]
    
    #--------- Cov: my method
    # B = nn.Dense(1, name='enc_cov', kernel_init=self.kernel_init)(z) # shape [b N d 1]
    # B = jnp.squeeze(B, axis=-1) # shape [b N d]
    # B_transp = jnp.swapaxes(B, -1, -2)
    # C = B_transp @ B # shape [b d d]
    
    
    
    #------------------------------
    # d = C.shape[-1]
    # std_ = jax.vmap(jnp.diag, in_axes=(0))(C)
    # std_ = std_.sum(-1, keepdims=True)
    # std = jnp.ones(d) * std_

    # rho_ij = 0.9 
    # R = jnp.eye(d) + rho_ij * (1 - jnp.eye(d))           

    # C = jax.vmap(jnp.outer, in_axes=(0))(std, std) * R
    # print(f'{C.shape} - C.shape out')
    #------------------------------
    
    return mu, sigma, C
  
  
class VAE2(nn.Module):
  """Full VAE model"""
  
  num_layers: int = 4
  num_layers_param1: int = 1
  emb_dim: int = 32
  num_heads: int = 4
  ffn_dim_factor: int = 4
  dropout_prob: float = 0.0
  
  num_s_samples: int = 10
  num_sdec_samples: int = 1
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  def setup(self):
    self.encoder = Encoder2(self.num_layers, self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)

  def __call__(self, x, z_rng, train=True):
    mu, sigma, C = self.encoder(x, train)
    enc_out = latent_space_samples(z_rng, mu, sigma, self.num_s_samples, self.num_sdec_samples)
    return mu, sigma, C, enc_out

  

#----- VAE (for Prior Ablation study): predicting sigma

class Encoder2_1(nn.Module):
  """VAE Encoder"""
  
  num_layers: int
  num_layers_param1: int
  emb_dim: int
  num_heads: int
  ffn_dim_factor: int
  dropout_prob: float
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  @nn.compact
  def __call__(self, x, train):
    print(f'---VAE2_1: encoder') 
      
    #--------- separate
    # z = TransformerEncoder(self.num_layers, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k] 
    # x = TransformerEncoder(self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(z, train=train, fresh=False) # shape [b N d k]
    # mu = nn.Dense(1, name='enc_mean', kernel_init=self.kernel_init)(x) 
    # mu = rearrange(mu, 'b N d 1 -> b N d')
    # sigma = jnp.ones_like(mu) * 0.1
    
    #--------- same
    z = TransformerEncoder(self.num_layers, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k]
    mu = nn.Dense(1, name='enc_mean', kernel_init=self.kernel_init)(z) 
    mu = rearrange(mu, 'b N d 1 -> b N d')
    
    sigma = jnp.exp(nn.Dense(1, name='enc_logvar', kernel_init=self.kernel_init)(z)) 
    sigma = rearrange(sigma, 'b N d 1 -> b N d')

    
    
    #--------- Cov: avici-style
    B = nn.Dense(self.emb_dim, name='enc_cov', kernel_init=self.kernel_init)(z) # shape [b N d k]
    B = jnp.max(B, axis=-3) # shape [b d k]
    B_transp = jnp.swapaxes(B, -1, -2)
    C = B @ B_transp # shape [b d d]
    
    #--------- Cov: my method
    # B = nn.Dense(1, name='enc_cov', kernel_init=self.kernel_init)(z) # shape [b N d 1]
    # B = jnp.squeeze(B, axis=-1) # shape [b N d]
    # B_transp = jnp.swapaxes(B, -1, -2)
    # C = B_transp @ B # shape [b d d]
    
    
    
    #------------------------------
    # d = C.shape[-1]
    # std_ = jax.vmap(jnp.diag, in_axes=(0))(C)
    # std_ = std_.sum(-1, keepdims=True)
    # std = jnp.ones(d) * std_

    # rho_ij = 0.9 
    # R = jnp.eye(d) + rho_ij * (1 - jnp.eye(d))           

    # C = jax.vmap(jnp.outer, in_axes=(0))(std, std) * R
    # print(f'{C.shape} - C.shape out')
    #------------------------------
    
    return mu, sigma, C
  
  
class VAE2_1(nn.Module):
  """Full VAE model"""
  
  num_layers: int = 4
  num_layers_param1: int = 1
  emb_dim: int = 32
  num_heads: int = 4
  ffn_dim_factor: int = 4
  dropout_prob: float = 0.0
  
  num_s_samples: int = 10
  num_sdec_samples: int = 1
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  def setup(self):
    self.encoder = Encoder2_1(self.num_layers, self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)

  def __call__(self, x, z_rng, train=True):
    mu, sigma, C = self.encoder(x, train)
    enc_out = latent_space_samples(z_rng, mu, sigma, self.num_s_samples, self.num_sdec_samples)
    return mu, sigma, C, enc_out





#----- VAE: for Poisson model

class Encoder3(nn.Module):
  """VAE Encoder"""
  
  num_layers: int
  num_layers_param1: int
  emb_dim: int
  num_heads: int
  ffn_dim_factor: int
  dropout_prob: float
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  @nn.compact
  def __call__(self, x, train):
    print(f'---VAE3: encoder') 
    N = x.shape[-2]
    d = x.shape[-1]
      
    #--------- separate
    # z = TransformerEncoder(self.num_layers, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k] 
    # x = TransformerEncoder(self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(z, train=train, fresh=False) # shape [b N d k]
    # mu = nn.Dense(1, name='enc_mean', kernel_init=self.kernel_init)(x) 
    # mu = rearrange(mu, 'b N d 1 -> b N d')
    # sigma = jnp.ones_like(mu) * 0.1
    
    #--------- same
    z = TransformerEncoder(self.num_layers, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k]
    mu = nn.Dense(1, name='enc_mean', kernel_init=self.kernel_init)(z) 
    mu = rearrange(mu, 'b N d 1 -> b N d')
    sigma = jnp.ones_like(mu) * 0.1
    
    
    #--------- pi (dropout prob)
    # z = nn.Dense(self.emb_dim, name='enc_k', kernel_init=self.kernel_init)(z) # shape [b N d k]
    pi = nn.Dense(1, name='enc_pi', kernel_init=self.kernel_init)(z) # shape [b N d]
    pi = rearrange(pi, 'b N d 1 -> b N d')
    pi = jax.nn.sigmoid(pi) 
    
    #--------- outl (outlier effect)
    # outl = nn.Dense(1, name='enc_outl', kernel_init=self.kernel_init)(z) # shape [b N d]
    # outl = rearrange(outl, 'b N d 1 -> b N d')
    # outl = jnp.sum(outl, axis=-2, keepdims=True) # shape [b 1 d]
    # outl = jnp.repeat(outl, N, axis=-2)
    
    #--------- lib (library size)
    # lib = nn.Dense(1, name='enc_lib', kernel_init=jax.nn.initializers.constant(1e-3))(z) # shape [b N d]
    # lib = nn.Dense(1, name='enc_lib', kernel_init=self.kernel_init)(z) # shape [b N d]
    # lib = rearrange(lib, 'b N d 1 -> b N d')
    # lib = jnp.exp(jnp.mean(lib, axis=-1, keepdims=True)) # shape [b N 1]
    # # lib = jax.nn.softplus(jnp.mean(lib, axis=-1, keepdims=True)) # shape [b N 1]
    # lib = jnp.repeat(lib, d, axis=-1)
    
    #--------- shift (shift)
    shift = nn.Dense(1, name='enc_shift', kernel_init=self.kernel_init)(z) # shape [b N d]
    shift = rearrange(shift, 'b N d 1 -> b N d')
    shift = jnp.mean(shift, axis=-2, keepdims=True) # shape [b 1 d]
    
    tech_noise_params = {}
    tech_noise_params['dropout_prob'] = pi
    # tech_noise_params['outliers'] = outl
    # tech_noise_params['lib'] = lib
    tech_noise_params['shift'] = shift

    return mu, sigma, tech_noise_params
  

class VAE3(nn.Module):
  """Full VAE model"""
  
  num_layers: int = 4
  num_layers_param1: int = 1
  emb_dim: int = 32
  num_heads: int = 4
  ffn_dim_factor: int = 4
  dropout_prob: float = 0.0
  
  num_s_samples: int = 10
  num_sdec_samples: int = 1
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  def setup(self):
    self.encoder = Encoder3(self.num_layers, self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)

  def __call__(self, x, z_rng, train=True):
    mu, sigma, tech_noise_params = self.encoder(x, train)
    enc_out = latent_space_samples(z_rng, mu, sigma, self.num_s_samples, self.num_sdec_samples) 

    return mu, sigma, tech_noise_params, enc_out


  



#----- VAE: for Poisson model, predicting pi with dec

class Encoder4(nn.Module):
  """VAE Encoder"""
  
  num_layers: int
  num_layers_param1: int
  emb_dim: int
  num_heads: int
  ffn_dim_factor: int
  dropout_prob: float
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  @nn.compact
  def __call__(self, x, train):
    print(f'---VAE4: encoder') 
    N = x.shape[-2]
    d = x.shape[-1]
      
    #--------- separate
    # z = TransformerEncoder(self.num_layers, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k] 
    # x = TransformerEncoder(self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(z, train=train, fresh=False) # shape [b N d k]
    # mu = nn.Dense(1, name='enc_mean', kernel_init=self.kernel_init)(x) 
    # mu = rearrange(mu, 'b N d 1 -> b N d')
    # sigma = jnp.ones_like(mu) * 0.1
    
    #--------- same
    z = TransformerEncoder(self.num_layers, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k]
    mu = nn.Dense(1, name='enc_mean', kernel_init=self.kernel_init)(z) 
    mu = rearrange(mu, 'b N d 1 -> b N d')
    sigma = jnp.ones_like(mu) * 0.1
    
    
    #--------- pi (dropout prob)
    # z = nn.Dense(self.emb_dim, name='enc_k', kernel_init=self.kernel_init)(z) # shape [b N d k]
    pi = nn.Dense(1, name='enc_pi', kernel_init=self.kernel_init)(z) # shape [b N d]
    pi = rearrange(pi, 'b N d 1 -> b N d')
    pi = jax.nn.sigmoid(pi) 
    
    #--------- outl (outlier effect)
    # outl = nn.Dense(1, name='enc_outl', kernel_init=self.kernel_init)(z) # shape [b N d]
    # outl = rearrange(outl, 'b N d 1 -> b N d')
    # outl = jnp.sum(outl, axis=-2, keepdims=True) # shape [b 1 d]
    # outl = jnp.repeat(outl, N, axis=-2)
    
    #--------- lib (library size)
    # lib = nn.Dense(1, name='enc_lib', kernel_init=jax.nn.initializers.constant(1e-3))(z) # shape [b N d]
    # lib = nn.Dense(1, name='enc_lib', kernel_init=self.kernel_init)(z) # shape [b N d]
    # lib = rearrange(lib, 'b N d 1 -> b N d')
    # lib = jnp.exp(jnp.mean(lib, axis=-1, keepdims=True)) # shape [b N 1]
    # # lib = jax.nn.softplus(jnp.mean(lib, axis=-1, keepdims=True)) # shape [b N 1]
    # lib = jnp.repeat(lib, d, axis=-1)
    
    #--------- shift (shift)
    shift = nn.Dense(1, name='enc_shift', kernel_init=self.kernel_init)(z) # shape [b N d]
    shift = rearrange(shift, 'b N d 1 -> b N d')
    shift = jnp.mean(shift, axis=-2, keepdims=True) # shape [b 1 d]
    
    tech_noise_params = {}
    # tech_noise_params['dropout_prob'] = pi
    # tech_noise_params['outliers'] = outl
    # tech_noise_params['lib'] = lib
    tech_noise_params['shift'] = shift

    return mu, sigma, tech_noise_params

class Decoder4(nn.Module):
  """VAE Decoder"""
  
  num_layers_param1: int
  emb_dim: int
  num_heads: int
  ffn_dim_factor: int
  dropout_prob: float
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  @nn.compact
  def __call__(self, x, train):
    print(f'---VAE4: decoder')
    
    z = TransformerEncoder(self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)(x, train=train) # shape [b N d k]
    
    #--------- pi (dropout prob)
    pi = nn.Dense(1, name='dec_pi', kernel_init=self.kernel_init)(z) # shape [b N d]
    pi = rearrange(pi, 'b N d 1 -> b N d')
    pi = jax.nn.sigmoid(pi)
    
    return pi
  
class VAE4(nn.Module):
  """Full VAE model"""
  
  num_layers: int = 4
  num_layers_param1: int = 1
  emb_dim: int = 32
  num_heads: int = 4
  ffn_dim_factor: int = 4
  dropout_prob: float = 0.0
  
  num_s_samples: int = 10
  num_sdec_samples: int = 1
  kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

  def setup(self):
    self.encoder = Encoder4(self.num_layers, self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)
    self.decoder = Decoder4(self.num_layers_param1, self.emb_dim, self.num_heads, self.ffn_dim_factor, self.dropout_prob, self.kernel_init)

  def __call__(self, x, z_rng, train=True):
    mu, sigma, tech_noise_params = self.encoder(x, train)
    enc_out = latent_space_samples(z_rng, mu, sigma, self.num_s_samples, self.num_sdec_samples) 
    pi = self.decoder(enc_out[0], train)
    
    tech_noise_params['dropout_prob'] = pi

    return mu, sigma, tech_noise_params, enc_out

