import jax
import jax.numpy as jnp
import orbax.checkpoint
from flax.training import orbax_utils
from jax.lib import xla_bridge
from functools import partial
from jax.scipy.stats import norm, multivariate_normal, poisson
import optax

from einops import rearrange

import wandb
import time
import os

# from functions.synthetic.distributions import Gaussian_jax, Uniform_jax
# from functions.synthetic.graph import ErdosRenyi_jax
# from functions.vae import *
# from functions.prior import *



def train(num_epochs, state, train_loader, rng, kl_params, prior, tag=None, epoch_save_weights=None):
    epoch_start = 0
    num_s_samples = kl_params['num_s_samples']
    
    for epoch in range(num_epochs):
        
        start_time = time.time()
        loss = 0
        kl = 0
        rec = 0
        
        for b in train_loader:
            batch = b[0]
            batch_signal = b[1]
            datasets= b[2]
            
            print('----------------- training inputs ---------------')
            print(f'{batch.shape} - batch shape into train_step()')
            print(f'{batch[0,0:6]}) - batch[0][0:6]')
            
            # batch_C = jnp.stack([dataset.C for dataset in datasets])
            # batch_W = jnp.stack([dataset.W for dataset in datasets])
            # batch_G = jnp.stack([dataset.G for dataset in datasets])
            # print(f'{batch_C} - true Cov')
            # print(f'{batch_W * batch_G} - true adjacency matrix')
            # print(f'{batch_G} - true G')
            

            print('----------------- training results ---------------')
            
            
            rng, rng_z, rng_dropout = jax.random.split(rng, 3)
            state, loss_b, kl_b, rec_b, log_q_sx, log_p_s, S_samples, mu, sigma, C, grads = train_step(state, batch, rng_z, rng_dropout, num_s_samples, prior)
            
            print(f'{C} - predicted cov')
            
            print(f'\t batch: loss: {loss_b:.3f} \t kl: {kl_b:.3f} \t rec: {rec_b:.3f}')
            print(f'\t - log_q_sx  {log_q_sx.mean():.4f}, [{log_q_sx.min():.4f}, {log_q_sx.max():.4f}]')
            print(f'\t - log_p_s \t {log_p_s.mean():.4f}, [{log_p_s.min():.4f}, {log_p_s.max():.4f}]')
            print(f'\t - S_samples [{S_samples.min():.4f}, {S_samples.max():.4f}] ')
            print(f'\t - mu \t\t [{mu.min():.4f}, {mu.max():.4f}] ')
            print(f'\t - sigma \t [{sigma.min():.4f}, {sigma.max():.4f}] ')
            

            print('--------------------------------------------------')
            
            loss += loss_b
            kl += kl_b
            rec += rec_b
    
            
        loss = loss / len(train_loader)
        kl = kl / len(train_loader)
        rec = rec / len(train_loader)
        
        end_time_train = time.time()
        end_time_eval = time.time()
        

        #----- wandb logs
        wandb.log({"epoch": epoch+1,
                   "train loss per epoch": jax.device_get(loss),
                   "rec per epoch": jax.device_get(rec),
                   "kl per epoch": jax.device_get(kl),
                   },
                  step=epoch+1) 
                
        #----- checkpints
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(state)
        options = orbax.checkpoint.CheckpointManagerOptions(
            save_interval_steps=epoch_save_weights, 
            max_to_keep=3, 
            create=True, 
            save_on_steps=None
            )

        checkpoint_path = os.path.abspath(f'checkpoints/')
        checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_path, orbax_checkpointer, options)
        checkpoint_manager.save((epoch+1), state, save_kwargs={'save_args': save_args})
        
        end_time_log = time.time()
        
        
        print(f'\n EPOCH {epoch + 1}/{num_epochs + epoch_start} \t train loss {loss:.3f} (per batch): rec {rec:.3f} + kl {kl:.3f} \t - training: {(end_time_train-start_time):.2f} sec  + (eval: {(end_time_eval-end_time_train):.2f} sec) + (logging: {(end_time_log-end_time_eval):.2f} sec) --> full epoch: {(end_time_log-start_time):.2f} sec \n')
                
        
        
#----------------------------------------------------------------------------------------------------------------------------------------------#

# @partial(jax.jit, static_argnames=['num_s_samples','prior'])
# def train_step(state, batch, rng_z, dropout_rng, num_s_samples, prior):
    
#     def loss_fn(params):
#         print(f'---train')
#         mu, sigma, enc_output, dec_output = state.apply_fn({'params': params}, batch, rng_z, train=True, rngs={'dropout': dropout_rng})

#         rec = reconstruction(batch, enc_output, dec_output)
#         kl, log_q_sx, log_p_s, S_samples, mu, sigma  = kl_divergence(mu, sigma, rng_z, num_s_samples, prior)
#         loss = rec + kl
        
#         return loss, (kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma)

#     # grads = jax.grad(loss_fn)(state.params)
#     (loss, (kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
#     return state.apply_gradients(grads=grads), loss, kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma, grads 


# def kl_divergence(mu, sigma, rng_z, num_s_samples, prior):
    
#     # #variational distribution (with distrax)
#     # q = distrax.Normal(mu, sigma)
#     # S_samples = q.sample(seed=rng_z, sample_shape=(num_s_samples,))
#     # log_q_sx = q.log_prob(S_samples).sum((-2,-1))
    
#     # print(f'{rng_z} - rng_z inside kl_divergence')
    
#     #variational distribution 
#     shape = (num_s_samples,) + sigma.shape # [s b N d]
#     S_samples = mu + sigma * jax.random.normal(rng_z, shape=shape)
#     log_q_sx = norm.logpdf(S_samples, mu, sigma).sum((-2,-1))
        
#     rng_prior, rng2 = jax.random.split(rng_z)
#     log_p_s = prior.log_prob(S_samples, rng_prior)
    
#     # print(f'{S_samples} - S_samples, shape: {S_samples.shape}')
    
#     kl = (log_q_sx - log_p_s).mean(-2)
#     kl = kl.mean()
    
#     # print(f'{norm.logpdf(S_samples, mu, sigma)} - log_q_sx_all')
    
#     return kl, log_q_sx, log_p_s, S_samples, mu, sigma    
  

# def reconstruction(batch, enc_output, dec_output):
#     N = batch.shape[-2]
    
#     dec_output = rearrange(dec_output, 's b d1 d2 -> s b 1 d1 d2')
#     dec_output = jnp.repeat(dec_output, N, axis=-3)
    
#     log_probs = multivariate_normal.logpdf(batch, mean=enc_output, cov=dec_output)

#     rec = (-1) * log_probs.sum((-1)) # output is torch.Size([s, b])
#     rec = rec.mean(-1) # mean over samples S1...SM ~ q(S), herec M=s=num_s_samples
#     rec = rec.mean() # mean over batches
    
#     return rec


################################################################################################################################################################################

@partial(jax.jit, static_argnames=['num_s_samples','prior'])
def train_step(state, batch, rng_z, dropout_rng, num_s_samples, prior):
    
    def loss_fn(params):
        print(f'---train')
        mu, sigma, C, enc_output = state.apply_fn({'params': params}, batch, rng_z, train=True, rngs={'dropout': dropout_rng})

        rec = reconstruction(batch, enc_output, C)
        kl, log_q_sx, log_p_s, S_samples, mu, sigma  = kl_divergence(mu, sigma, rng_z, num_s_samples, prior)
        loss = rec + kl
        
        return loss, (kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma, C)

    # grads = jax.grad(loss_fn)(state.params)
    (loss, (kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma, C)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma, C, grads 


def reconstruction(batch, enc_output, C):
    # batch of shape [b N d]
    # enc_output of shape [s b N d]
    # C of shape [b d d]
    
    print('----- reconstruction (NLL) -----')
    print(f'{C.shape} - C initial shape')
    
    N = batch.shape[-2]
    srec = enc_output.shape[-4]
    
    #----- VAE2
    C = rearrange(C, 'b d1 d2 -> 1 b 1 d1 d2')
    C = jnp.repeat(C, N, axis=-3) # shape [s b N d d]
    C = jnp.repeat(C, srec, axis=-5) # shape [s b N d d]
    
    #----- VAE
    # C = rearrange(C, 'sdec b d1 d2 -> sdec b 1 d1 d2')
    # C = jnp.repeat(C, N, axis=-3) # shape [s b N d d]
    
    print(f'{C.shape} - C')
    print(f'{enc_output.shape} - enc_output = S')
    
    # print('---reconstruction')
    # print(f'{batch} - batch, {batch.shape}')
    # print(f'{enc_output} - enc_output, {enc_output.shape}')
    
    log_probs = multivariate_normal.logpdf(batch, mean=enc_output, cov=C) # shape [s b N]
    # log_probs = poisson.logpmf(batch, mu=enc_output)
    
    print(f'{log_probs.shape} - log_probs')
    
    # (1, 2, 10, 3) - enc_output = S    for [s b N d]
    # (   2, 10, 3, 3) - C              for   [b N d d]
    # (1, 2, 10) - log_probs            for [s b N]

    rec = (-1) * log_probs.sum((-1)) # output is [b]
    rec = rec.mean(-1) # mean over samples S1...SM ~ q(S), herec M=s=num_s_samples
    rec = rec.mean() # mean over batches
    
    return rec

#-------KL with our prior
def kl_divergence(mu, sigma, rng_z, num_s_samples, prior):
    
    print('----- KL divergence -----')
    print(f'{mu.shape} - mu in q(S|D)')
    print(f'{sigma.shape} - sigma in q(S|D)')
    print('-------------------------')
    
    #variational distribution 
    shape = (num_s_samples,) + sigma.shape # [s b N d]
    S_samples = mu + sigma * jax.random.normal(rng_z, shape=shape)
    log_q_sx = norm.logpdf(S_samples, mu, sigma).sum((-2,-1))
        
    rng_prior, rng2 = jax.random.split(rng_z)
    log_p_s = prior.log_prob(S_samples, rng_prior)
    
    # print(f'{S_samples} - S_samples, shape: {S_samples.shape}')
    
    kl = (log_q_sx - log_p_s).mean(-2)
    kl = kl.mean()
    
    # print(f'{norm.logpdf(S_samples, mu, sigma)} - log_q_sx_all')
    
    return kl, log_q_sx, log_p_s, S_samples, mu, sigma   


#-------for testing isptropic Gaussian prior
# def kl_divergence(mu, sigma, rng_z, num_s_samples, prior):
    
#     print('----- KL divergence (isotropic Gaussian) -----')
#     print(f'{mu.shape} - mu shape')
#     print(f'{sigma.shape} - sigma shape')
    
#     # ----- KL divergence -----
#     # (10, 500, 4) - mu shape
#     # (10, 500, 4) - mu
#     # -------------------------
    
#     kl = 1/2 * (sigma ** 2 + mu ** 2 - 1 - jnp.log(sigma ** 2)).sum(axis=-1) # sum over axis d and the output has shape [b N]
#     print(f'{kl.shape} - before summing over N and averaging over b')
#     kl = jnp.sum(kl, axis=-1) # sum over axis N
#     kl = jnp.mean(kl)
#     print('-------------------------')
    
#     log_q_sx, log_p_s, S_samples = 0, 0, 0
    
#     return kl, log_q_sx, log_p_s, S_samples, mu, sigma  


################################################################################################################################################################################

#----------------------------------------------------------------------------------------------------------------------------------------------#       
#---- here train_step and kl are with minimal outputs

# @partial(jax.jit, static_argnames=['num_s_samples','prior'])
# def train_step(state, batch, rng_z, dropout_rng, num_s_samples, prior):
    
#     def loss_fn(params):
#         print(f'---train')
#         mu, sigma, enc_output, dec_output = state.apply_fn({'params': params}, batch, rng_z, train=True, rngs={'dropout': dropout_rng})

#         rec = reconstruction(batch, enc_output, dec_output)
#         kl = kl_divergence(mu, sigma, rng_z, num_s_samples, prior)
#         loss = rec + kl
        
#         return loss, (kl, rec)

#     # grads = jax.grad(loss_fn)(state.params)
#     (loss, (kl, rec)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
#     return state.apply_gradients(grads=grads), loss, kl, rec


# def kl_divergence(mu, sigma, rng_z, num_s_samples, prior):
    
#     # #variational distribution (with distrax)
#     # q = distrax.Normal(mu, sigma)
#     # S_samples = q.sample(seed=rng_z, sample_shape=(num_s_samples,))
#     # log_q_sx = q.log_prob(S_samples).sum((-2,-1))
    
#     #variational distribution 
#     shape = (num_s_samples,) + sigma.shape
#     S_samples = mu + sigma * jax.random.normal(rng_z, shape=shape)
#     log_q_sx = norm.logpdf(S_samples, mu, sigma).sum((-2,-1))
        
#     rng_prior, rng2 = jax.random.split(rng_z)
    
#     log_p_s = prior.log_prob(S_samples, rng_prior)
    
#     # print(f'{S_samples} - S_samples, shape: {S_samples.shape}')
    
#     kl = (log_q_sx - log_p_s).mean(-2)
#     kl = kl.mean()
    
#     return kl

# @staticmethod
# def l2_squared(tree):
#     leaves, _ = jax.tree_util.tree_flatten(tree)
#     return jnp.sum(jnp.array([jnp.vdot(x, x) for x in leaves]))


# # @partial(jax.jit, static_argnames=['num_s_samples','prior'])
# def eval_step(state, batch, rng_z, dropout_rng, num_s_samples, prior):
    
#     print(f'---eval')    
#     mu, sigma, enc_output, dec_output = state.apply_fn({'params': state.params}, batch, rng_z, train=False, rngs={'dropout': dropout_rng})
    
#     rec = reconstruction(batch, enc_output, dec_output)
#     kl = kl_divergence(mu, sigma, rng_z, num_s_samples, prior)
    
#     loss = rec + kl
#     # loss = rec + kl + 0.01 * l2_squared(state.params['decoder'])
#     # loss = rec + kl + 0.001 * l2_squared(state.params['decoder'])
    
#     print(f'eval kl: {kl} \neval rec: {rec}')

#     return loss, kl, rec


# @partial(jax.jit, static_argnames=['num_s_samples','prior'])
# def eval_step(state, batch, rng_z, dropout_rng, num_s_samples, prior):
    
#     print(f'---eval')    
#     mu, sigma, enc_output, dec_output = state.apply_fn({'params': state.params}, batch, rng_z, train=False, rngs={'dropout': dropout_rng})
    
#     rec = reconstruction(batch, enc_output, dec_output)
#     kl = kl_divergence(mu, sigma, rng_z, num_s_samples, prior)
#     loss = rec + kl

    
#     return loss, kl, rec

#----------------------------------------------------------------------------------------------------------------------------------------------#


#------------------ with Wasserstein distance------------------
# @partial(jax.jit, static_argnames=['num_s_samples','prior','epoch'])
# def eval_step(state, batch, rng_z, dropout_rng, num_s_samples, prior, batch_signal, epoch):
    
#     print(f'---eval')    
#     mu, sigma, enc_output, dec_output = state.apply_fn({'params': state.params}, batch, rng_z, train=False, rngs={'dropout': dropout_rng})
    
#     rec = reconstruction(batch, enc_output, dec_output)
#     kl = kl_divergence(mu, sigma, rng_z, num_s_samples, prior)
#     loss = rec + kl
    
#     if ((epoch+1) % 1000) == 0:
#         epsilon = 0.1
#         wasserstein = jax.vmap(wasserstein_fun, in_axes=(0, 0, None))(enc_output[0], batch_signal, epsilon)
#     else:
#         wasserstein = 0
    
#     return loss, kl, rec, wasserstein.mean()

# @jax.jit
# def wasserstein_fun(target_x, target_y, epsilon):
#     assert target_x.ndim == target_y.ndim == 2 and target_x.shape[-1] == target_y.shape[-1]
#     a = jnp.ones(len(target_x)) / len(target_x)
#     b = jnp.ones(len(target_y)) / len(target_y)

#     out_xy = sinkhorn_divergence(
#         pointcloud.PointCloud,
#         target_x,
#         target_y,
#         a=a,
#         b=b,
#         epsilon=epsilon,
#         symmetric_sinkhorn=False,
#     )
#     converged = jnp.all(jnp.array(out_xy.converged))
#     return jnp.where(converged, out_xy.divergence, jnp.nan)
#------------------------------------------------------------------------


# def eval_f(state, batch, z_rng, dropout_rng):
    
#     def eval_model(vae):
#         mu, sigma, enc_output, dec_output = vae(batch, z_rng, train=True, rngs={'dropout': dropout_rng})
#         rec = reconstruction(batch, enc_output, dec_output)
#         kl = kl_divergence(mu, sigma, z_rng)
#         loss = rec + kl
#         return loss, kl, rec

#     return nn.apply(eval_model, state.apply_fn)({'params': state.params})







# def train(num_epochs, state, train_loader, rng, prior_kwargs, latents):
#     epoch_start = 0
#     for epoch in range(num_epochs):
#         for ds in train_loader:
#             batch = ds[0]
#             #--------
#             batch = jnp.array(batch.numpy())
            
#             datasets = ds[1]
#             newline = "\n"
#             # print(f'{  newline.join(f" G in dataset {i+1}: {datasets[i].G}" for i in range(len(ds[1]))) }')
#             # print(f'{  newline.join(f" W in dataset {i+1}: {datasets[i].W}" for i in range(len(ds[1]))) }')
#             print(f'{  newline.join(f" w_ij in dataset {i+1}: {0 if ( ((datasets[i].W * datasets[i].G) != 0).sum().item() == 0 ) else (  (datasets[i].W * datasets[i].G)[(datasets[i].W * datasets[i].G) != 0].item()  ) }" for i in range(len(ds[1]))) }')
#             #--------
#             rng, key = random.split(rng)
#             state = train_step(state, batch, key, latents)
        
#         loss, kl, rec = eval_f(state.params, batch, key, latents)
        
#         print(f'\n EPOCH {epoch + 1}/{num_epochs + epoch_start} \t train loss {loss:.3f} (per batch): rec {rec:.3f} + kl {kl:.3f}')


# @jax.jit
# def train_step(state, batch, z_rng, latents):
    
#     def loss_fn(params):
#         mu, sigma, enc_output, dec_output = vae(latents).apply({'params': params}, batch, z_rng)

#         rec = reconstruction(batch, enc_output, dec_output)
#         kl = kl_divergence(mu, sigma, z_rng)
#         loss = rec + kl
        
#         return loss

#     grads = jax.grad(loss_fn)(state.params)
#     return state.apply_gradients(grads=grads)


# def kl_divergence(mu, sigma, z_rng):
#     num_s_samples = 10
    
#     q = distrax.Normal(mu, sigma)
#     S_samples = q.sample(seed=z_rng, sample_shape=(num_s_samples,))
    
#     prior = distrax.Normal(jnp.zeros_like(mu), jnp.ones_like(sigma))
    
#     log_q_sx = q.log_prob(S_samples).sum((-2,-1))
#     log_p_s = prior.log_prob(S_samples).sum((-2,-1))
    
#     kl = (log_q_sx - log_p_s).mean(-2)
#     kl = kl.mean()
    
#     return kl

# def reconstruction(batch, enc_output, dec_output):
#     N = 500
    
#     dec_output = rearrange(dec_output, 'b d1 d2 -> b 1 d1 d2')
#     dec_output = jnp.repeat(dec_output, N, axis=-3)
    
#     log_probs = jax.scipy.stats.multivariate_normal.logpdf(batch, mean=enc_output, cov=dec_output)

#     rec = (-1) * log_probs.sum((-1)) # output is torch.Size([s, b])
#     rec = rec.mean(-1) # mean over samples S1...SM ~ q(S), herec M=s=num_s_samples
#     rec = rec.mean() # mean over batches
    
#     return rec

# def eval_f(params, batch, z_rng, latents):
    
#     def eval_model(vae):
#         mu, sigma, enc_output, dec_output = vae(batch, z_rng)
#         rec = reconstruction(batch, enc_output, dec_output)
#         kl = kl_divergence(mu, sigma, z_rng)
#         loss = rec + kl
#         return loss, kl, rec

#     return nn.apply(eval_model, vae(latents))({'params': params})