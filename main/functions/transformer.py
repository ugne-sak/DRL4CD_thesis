import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Sequence, Callable, NamedTuple, Optional, Tuple
from einops import rearrange


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

class TransformerEncoderLayer(nn.Module):
    emb_dim : int  
    num_heads : int
    ffn_dim_factor : int
    dropout_prob : float
    kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] =  nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal() 
    
    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.emb_dim, kernel_init=self.kernel_init)
        
        self.norm1q = nn.LayerNorm()
        self.norm1kv = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        
        self.dropout1 = nn.Dropout(self.dropout_prob)
        self.dropout2 = nn.Dropout(self.dropout_prob)
        
        self.ffn = nn.Sequential([
            nn.Dense(self.emb_dim*self.ffn_dim_factor, kernel_init=self.kernel_init),
            nn.relu,
            nn.Dense(self.emb_dim, kernel_init=self.kernel_init)
        ])


    def __call__(self, x, train=True):
        
        q_in = self.norm1q(x) # norm before attn
        kv_in = self.norm1kv(x) # norm before attn
        
        x_attn = self.attn(q_in,kv_in)
        x = x + self.dropout1(x_attn, deterministic=not train) # add
        
        x_in = self.norm2(x) # norm before ffn
        x_ffn = self.ffn(x_in)
        x = x + self.dropout2(x_ffn, deterministic=not train) # add 
        
        return x

class TransformerEncoder(nn.Module):
    
    num_layers: int
    emb_dim: int
    num_heads: int
    ffn_dim_factor: int
    dropout_prob: float
    kernel_init : Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init # jax.nn.initializers.lecun_normal()

    def setup(self):
        self.linear_in = nn.Dense(self.emb_dim, kernel_init=self.kernel_init)
        self.encoder_layers = [TransformerEncoderLayer(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            ffn_dim_factor=self.ffn_dim_factor,
            dropout_prob=self.dropout_prob,
            kernel_init=self.kernel_init
        ) for _ in range(self.num_layers*2)]
        self.norm3 = nn.LayerNorm()

    def __call__(self, x, train=True, fresh=True):
        print(f'x shape into encoder layer: {x.shape} \t continue: {not fresh}')
        
        if fresh:
            x = jnp.expand_dims(x, axis=-1)
            x = self.linear_in(x)
        elif x.shape[-1] != self.emb_dim:
            x = self.linear_in(x)

        for encoder_layer in self.encoder_layers:
            print(f'x shape into encoder sublayer: {x.shape}')
            x = encoder_layer(x, train=train)
            x = jnp.swapaxes(x, -3, -2)
        
        x = self.norm3(x)
        
        print(f'x shape out from encoder layer: {x.shape} - for [b N d k] or [s b N d k]')

        return x
