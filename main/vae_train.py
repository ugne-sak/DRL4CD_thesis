import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
from flax.training import train_state
import optax
import yaml
import numpy as np
import time
import datetime 
import wandb
import pprint
import hydra
import os
from omegaconf import OmegaConf
import traceback
import sys
import importlib
import shutil

import functions.vae as vae_module
from functions.prior import LatentPrior
from functions.synthetic.data_jax import SyntheticDataGenerator, dataset_loader
from functions.utils import parse_config

# os.environ["JAX_LOG_LEVEL"] = "ERROR" # environment variable to exclude JAX informational warnings
import logging
logging.getLogger('jax').setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path='config', config_name='config_test')
def main(config):
    # this is needed to get all errors logged into .err files
    # solution taken from https://github.com/facebookresearch/hydra/issues/2664#issuecomment-1857695600
    
    try:
        actual_main(config)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()
        
def actual_main(config):
    
    #----- run info
    tag = config["setup"]["tag"]
    
    print(f'\n time: {datetime.datetime.now()}')
    print(f' tag: {tag}')
    print(f' Working directory : {os.getcwd()}')
    print(f' Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir} \n')
    
    
    cfg_file = hydra.core.hydra_config.HydraConfig.get().job.config_name
    print(cfg_file)
    print(hydra.utils.get_original_cwd())
    print(f'{hydra.utils.get_original_cwd()}/config/{cfg_file}\n')
    
    os.makedirs(os.path.dirname(f'config_file/{cfg_file}'), exist_ok=True)
    shutil.copyfile(f'{hydra.utils.get_original_cwd()}/config/{cfg_file}', f'config_file/{cfg_file}')
    
    #----- wandb
    wandb.init(
        **config['wandb'],
        config=OmegaConf.to_container(config, resolve=True)
        )
    
    print(f'\ntime: {datetime.datetime.now()}')
    print(f'tag: {tag}')
    
    #----- device info
    xla_backend_type = xla_bridge.get_backend().platform
    print(f"XLA backend type (device): {xla_backend_type}")
    
    if xla_backend_type == "gpu":
        for idx, device in enumerate(xla_bridge.get_backend().devices()):
            gpu_type = "Active GPU" if idx == 0 else "GPU"
            print(f"   {gpu_type} index: {device.id}")
            print(f"   {gpu_type} name:  {device.device_kind} \n")
    
    #----- rngs
    seed = config["train"]["seed"]
    print(f'\nseed: {seed}')
     
     
    #----------------- setup -----------------#
    
    #----- dataloader
    rng = jax.random.PRNGKey(seed)
    rng, subrng = jax.random.split(rng)
    
    rng_np = np.random.default_rng(np.random.SeedSequence(entropy=seed))

    mechanism = config['dataset_generator']['mechanism']
    print(f'mechanism: {mechanism}')

    if mechanism == 'linearGBN':
        mechanism_params = parse_config(config['linearGBN_params'], config_key='mechanism_params')
        technical_noise_params = parse_config(config['linearGBN_params'], config_key='technical_noise_params')
        graph_distr = parse_config(config, config_key='graph')['graph_distr']
        add_technical_noise_params = None

    if mechanism == 'grn_sergio':
        mechanism_params = parse_config(config['grn_sergio_params'], config_key='mechanism_params')
        technical_noise_params = parse_config(config['grn_sergio_params'], config_key='technical_noise_params')
        graph_distr = parse_config(config, config_key='graph')['graph_distr']
        add_technical_noise_params = None

    if mechanism == 'dual':
        mechanism_params = parse_config(config['linearGBN_params'], config_key='mechanism_params')
        technical_noise_params = parse_config(config['dual_params'], config_key='technical_noise_params')
        graph_distr = parse_config(config, config_key='graph')['graph_distr']
        add_technical_noise_params = None

    print(f'\nmechanism_params:\n{pprint.pformat(mechanism_params, depth=1)}')
    print(f'\ntechnical_noise_params:\n{pprint.pformat(technical_noise_params, depth=1)}')
    print(f'\ngraph_distr:\n{pprint.pformat(graph_distr, depth=1)}')
    
    
    data_generator = SyntheticDataGenerator(mechanism, mechanism_params, technical_noise_params, graph_distr, add_technical_noise_params=add_technical_noise_params)

    dataset_params = parse_config(config, config_key='dataset_params')
    dataset_params['rng_np'] = rng_np

    train_loader = dataset_loader(subrng, data_generator, dataset_params=dataset_params, meta_size=config.data_loader.meta_size, batch_size=config.data_loader.batch_size)

    
    
    #----- prior
    prior_kwargs = parse_config(config, config_key='prior')
    prior = LatentPrior(**prior_kwargs)
    
    print(f'\nprior params:\n{pprint.pformat(prior_kwargs, depth=1)}')
    
    #----- params for KL divergence
    kl_params = dict(
        num_s_samples=config.VAE_params.num_s_samples
    )
    
    
    
    #----------------- model -----------------#
    
    print(f'---init')
    
    def init_vae(vae_class, vae_params, rng, kernel_init=None):
    
        if kernel_init is not None:
            vae = vae_class(kernel_init=kernel_init, **vae_params)
        else:
            vae = vae_class(**vae_params)
        
        N, d = 50, 2
        init_data = jnp.ones((1, N, d), jnp.float32)
        rng, init_rng, init_rng_dropout, init_rng_z = jax.random.split(rng, 4)
        params = vae.init({'params': init_rng, 'dropout': init_rng_dropout}, init_data, init_rng_z, train=True)['params']
        
        return vae, params
    
    
    vae_class_name = config.VAE  # e.g., "VAE3"
    vae_class = getattr(vae_module, vae_class_name)     
    kernel_init = jax.nn.initializers.he_uniform()
    vae_params = parse_config(config, config_key='VAE_params')
    
    rng, rng_init = jax.random.split(rng, 2)
    vae, params = init_vae(vae_class, vae_params, rng_init, kernel_init)
    
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"\nUsing model with {param_count:,} parameters ({(param_count)/1000/1000:.{2}f} M parameters) \n")
    
    
    
    #----------------- training -----------------#

    num_epochs = config['train']['num_epochs']
    lr = float(config['train']['lr'])
    weight_decay = float(config['train']['weight_decay'])
    lamb_b1 = float(config['train']['lamb_b1'])
    lamb_b2 = float(config['train']['lamb_b2'])

    print(f'lr: {lr}')
    print(f'lamb_b1: {lamb_b1}')
    print(f'lamb_b2: {lamb_b2}')
    print(f'weight_decay: {weight_decay}')
    
    factor = int(config.data_loader.meta_size / config.data_loader.batch_size)
    print(f'factor: {factor} - every_k_schedule')
    
    optimizer = optax.lamb(lr, b1=lamb_b1, b2=lamb_b2, weight_decay=weight_decay)
    
    # optimizer = optax.chain(
    #     # optax.clip_by_global_norm(1),
    #     # optax.scale_by_schedule(optax.linear_onecycle_schedule(num_epochs, peak_value=lr, pct_start=0.3, div_factor=50)),
    #     optax.lamb(lr, b1=lamb_b1, b2=lamb_b2, weight_decay=weight_decay),
    # )
    
    # lr_schedule = optax.linear_onecycle_schedule(num_epochs, peak_value=lr, pct_start=0.3, div_factor=25)
    # lr_schedule = optax.warmup_exponential_decay_schedule(init_value=5e-3, peak_value=lr, warmup_steps=int(num_epochs*0.04), transition_steps=num_epochs, decay_rate=0.5, transition_begin=int(num_epochs*0.3))
    # optimizer = optax.lamb(lr_schedule, b1=lamb_b1, b2=lamb_b2, weight_decay=weight_decay)
    
    state = train_state.TrainState.create(
        apply_fn = vae.apply,
        params = params,
        # tx = optax.lamb(lr, weight_decay=weight_decay)
        tx = optax.MultiSteps(optimizer, every_k_schedule=1*factor)
    )
    
    #----- train function
    train_module_name = config.train_module  
    train_module = importlib.import_module(f"functions.{train_module_name}")
    train = train_module.train
    
    start_time = time.time()
    
    train(num_epochs, state, train_loader, rng, kl_params, prior, tag=config.setup.tag, epoch_save_weights=config.train.epoch_save_weights)
    
    end_time = time.time()
    print(f"training time: {str(datetime.timedelta(seconds=(end_time - start_time)))} seconds \n")

    wandb.finish()
    
    
    

if __name__ == "__main__":
    
    main()
 




#----------------------------   
# python vae_train_dual.py --config_file config/config_test.yaml 

# hydra
# python vae_train_dual.py -cn config_test.yaml 

# environment
# source ../.venv/env_test/bin/activate