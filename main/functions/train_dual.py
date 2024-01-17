import jax
import jax.numpy as jnp
import orbax.checkpoint
from flax.training import orbax_utils
from functools import partial
from jax.scipy.stats import norm, multivariate_normal, poisson

import wandb
import time
import os




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
            print(f'{jnp.count_nonzero(batch[0] == 0) / (batch[0].shape[-1] * batch[0].shape[-2])} - empirical proportion of zeros in batch[0]')
            print(f'{jnp.count_nonzero(batch[1] == 0) / (batch[1].shape[-1] * batch[1].shape[-2])} - empirical proportion of zeros in batch[1]')
            
            # print(f'{jnp.round(1.0 - datasets[0].non_drop_prob[0:6],1)} - true pi[0][0:6] in batch[0]')
            # batch_pi = jnp.stack([dataset.non_drop_prob for dataset in datasets])
            # print(f'{batch_pi.shape} - batch_pi.shape')
            
            # batch_C = jnp.stack([dataset.C for dataset in datasets])
            # batch_W = jnp.stack([dataset.W for dataset in datasets])
            # batch_G = jnp.stack([dataset.G for dataset in datasets])
            # print(f'{batch_C} - true Cov')
            # print(f'{batch_W * batch_G} - true adjacency matrix')
            # print(f'{batch_G} - true G')
            
            
            print('----------------- training results ---------------')
            
            
            rng, rng_z, rng_dropout = jax.random.split(rng, 3)
            state, loss_b, kl_b, rec_b, log_q_sx, log_p_s, S_samples, mu, sigma, tech_noise_params, grads, pi = train_step(state, batch, rng_z, rng_dropout, num_s_samples, prior) # TODO: batch_C only for sanity check
            
            # print(f'{jnp.round(tech_noise_params["dropout_prob"][0,0:6],1)} - predicted pi[0][0:6] ')
            # print(f'{jnp.round(tech_noise_params["dropout_prob"][0,0:6],3)} - predicted pi[0][0:6] ')
            print(f'{jnp.round(tech_noise_params["shift"][0],1)} - predicted shift[0]')
            print(f'{jnp.round(pi[0,0,0:6],1)} - computed pi[0][0:6], shape {pi.shape}')
            
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



@partial(jax.jit, static_argnames=['num_s_samples','prior'])
def train_step(state, batch, rng_z, dropout_rng, num_s_samples, prior):
    
    def loss_fn(params):
        print(f'---train')
        mu, sigma, tech_noise_params, enc_output = state.apply_fn({'params': params}, batch, rng_z, train=True, rngs={'dropout': dropout_rng})
        
        rec, pi = reconstruction(batch, enc_output, tech_noise_params)
        kl, log_q_sx, log_p_s, S_samples, mu, sigma  = kl_divergence(mu, sigma, rng_z, num_s_samples, prior)
        loss = 20*rec + kl #----------------------------------- TODO: modify if needed (for poisson model)
        
        return loss, (kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma, tech_noise_params, pi)

    # grads = jax.grad(loss_fn)(state.params)
    (loss, (kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma, tech_noise_params, pi)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, kl, rec, log_q_sx, log_p_s, S_samples, mu, sigma, tech_noise_params, grads, pi 


def reconstruction(batch, enc_output, tech_noise_params):
    """
        computes MC approximation of reconstruction loss given by the expected NLL of the measurement noise model p(D|S) 
        
        Args:
            batch: array of shape [b N d]
            enc_output: array of shape [s b N d] - unexponentiated sample!
            tech_noise_params: dict that contains predicted measurement noise parameters such as pi of shape [b N d] or shift of shape [b N d]
    """
    
    pi = tech_noise_params['dropout_prob']
    # outliers = tech_noise_params['outliers']
    # lib = tech_noise_params['lib'] # shape should be [b N d]
    shift = tech_noise_params['shift']
    
    print('----- reconstruction (NLL) -----')
    print(f'{batch.shape} - batch = D')
    print(f'{enc_output.shape} - enc_output = S')
    print(f'{pi.shape} - pi')
    print(f'{shift.shape} - shift')
    
    mean = jax.nn.softplus(enc_output + shift)
    # mean = jax.nn.softplus(enc_output)
    
    
    #-------------------- for computing pi as in SERGIO (alternative: predict pi with decoder) --------------------#
    
    # rng, rng_drop = jax.random.split(rng_1)
    # binary_ind, prob_ber = dropout_indicator(rng_drop, mean, shape=8, percentile=90)
    # pi = 1.0 - prob_ber # because dropout_indicator produces mask which indicates non-dropout instances with prob_ber denoting probability for NOT being a dropout
    #pi shape [b s N d]
    
    #--------------------------------------------------------------------------------------------------------------#
    
    
    # log_probs = poisson.logpmf(batch, mu=mean) #----------------------------------- TODO: uncomment for poisson model
    log_probs = zero_inflated_poisson_logpmf(batch, mean, pi) #----------------------------------- TODO: uncomment for zero-inflated poisson model
    # log_probs = zero_inflated_poisson_logpmf(batch_raw, mean, pi) #----------------------------------- TODO: uncomment for zero-inflated poisson model with predicting lib
    # log_probs = zero_inflated_poisson_logpmf(batch, outliers*enc_output, pi) #----------------------------------- TODO: uncomment for zero-inflated poisson model with predicting outliers
    
    print(f'{log_probs.shape} - log_probs') # [s b N d]

    rec = (-1) * log_probs.sum((-1, -2)) # output is [s b]
    rec = rec.mean(-2) # mean over samples S1...SM ~ q(S), herec M=s=num_s_samples
    rec = rec.mean() # mean over batches
    
    return rec, pi


def kl_divergence(mu, sigma, rng_z, num_s_samples, prior):
    """
        computes MC approximation of KL divergence between the approximate posterior q(S) ant true signal prior p(S) 
        
        Args:
            mu: mean of q(S), matrix of shape [s b N d], output of the encoder NN (predicted)
            sig: std of q(S), matrix of shape [s b N d], output of the encoder NN (fixed in our implementation but can be predicted as well)
    """
    
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
    
    kl = (log_q_sx - log_p_s).mean(-2)
    kl = kl.mean()
    
    return kl, log_q_sx, log_p_s, S_samples, mu, sigma   




"""---------- ZIP model ----------"""

def zero_inflated_poisson_logpmf(x, rate, pi):
    """
    Computes log PMF of Zero-Inflated Poisson distribution.
    
    Args:
        pi: probability of x being 0
    """
    log_prob_zero = jnp.log(pi + (1.0 - pi) * jnp.exp(-rate))
    log_prob_non_zero = poisson.logpmf(x, rate) + jnp.log((1.0 - pi)) 
    log_probs = jnp.where(x == 0, log_prob_zero, log_prob_non_zero)
    return log_probs 



"""---------- SERGIO-type technical noise ----------"""

def dropout_indicator(rng_key, scData, shape=1, percentile=65):
    """
    Similar to the Splat package.

    Input:
    scData can be the output of a simulator or any refined version of it
    (e.g., with technical noise).

    shape: the shape of the logistic function.

    percentile: the mid-point of logistic functions is set to the given percentile
    of the input scData.

    Returns: jnp.array containing binary indicators showing dropouts.
    """
    scData = jnp.array(scData)
    scData_log = jnp.log(scData + 1)
    log_mid_point = jnp.percentile(scData_log, percentile)
    prob_ber = 1 / (1 + jnp.exp(-1 * shape * (scData_log - log_mid_point)))

    binary_ind = jax.random.bernoulli(rng_key, prob_ber)

    return binary_ind, prob_ber