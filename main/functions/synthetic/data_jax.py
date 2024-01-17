import torch
from torch.utils.data import Dataset, DataLoader

from functions.synthetic.linear import *
from functions.synthetic.gene import GRNSergio, GRNSergio_np, GRNSergio_np2
from functions.utils import *

import jax
import jax.numpy as jnp
import numpy as np
import time



def apply_technical_noise(key, data, technical_noise_params):
    
    additional_noise_distr = technical_noise_params['additional_noise_distr']
    additional_noise_std_distr = technical_noise_params['additional_noise_std_distr']   

    N = data.shape[-2]
    d = data.shape[-1]
    
    #------correlated noise, additive
    key, subkey = jax.random.split(key)
    scale = additional_noise_std_distr(subkey, shape=())
    
    std = jnp.ones([1,d]) * scale
    rho_ij = 0.9 
    R = jnp.eye(d) + rho_ij * (1 - jnp.eye(d))       
    
    C = jnp.outer(std, std) * R
    
    mu = jnp.zeros(C.shape[0])
    key, subkey = jax.random.split(key)
    additional_noise_matrix = jax.random.multivariate_normal(key, mu, C, shape=(N,))
    
    dataset_noisy = data + additional_noise_matrix
    dataset_noisy.C = C

    return dataset_noisy


def apply_technical_noise2(key, data, technical_noise_params):
    
    additional_noise_distr = technical_noise_params['additional_noise_distr']
    additional_noise_std_distr = technical_noise_params['additional_noise_std_distr']   

    N = data.shape[-2]
    d = data.shape[-1]
    
    #------correlated noise, additive
    scale = additional_noise_std_distr(key)
    
    std = np.ones([1,d]) * scale
    rho_ij = 0.9 
    R = np.eye(d) + rho_ij * (1 - np.eye(d))           
    
    C = np.outer(std, std) * R
    
    mu = np.zeros(C.shape[0])
    additional_noise_matrix = np.random.multivariate_normal(mu, C, size=(N,))
    
    dataset_noisy = data + additional_noise_matrix
    
    dataset_noisy = torch.tensor(dataset_noisy)
    dataset_noisy.C = torch.tensor(C)

    return dataset_noisy


def apply_sergio_technical_noise(rng_np, data, technical_noise_params):
    
    # print(technical_noise_params)
    
    #---shift data
    shift_distr = technical_noise_params['shift_distr']
    shift = shift_distr(rng_np, data.shape[-1])
    shift = np.repeat(shift[np.newaxis, :], data.shape[-2], axis=0)
    # print(f'{shift} = shift, shape {shift.shape}')
    # print(f'shift, shape {shift.shape}')
    data = data + shift
    
    #---transform to get positive values
    data = np.logaddexp(0,data) # calculates log(exp(x1) + exp(x2)) so we have softplus log(1 + exp(x)) when x1=0
    
    # shift = rng_np
    # data = data + shift
    
    # # add technical noise (assume 1 experimental setting for wild type and KO experiments)
    # # 1) outlier genes
    # if self.add_outlier_effect:
    #     expr = sim.outlier_effect(expr, outlier_prob, outlier_mean, outlier_scale)

    # # 2) library size
    # if self.add_lib_size_effect:
    #     _, expr = sim.lib_size_effect(expr, lib_size_mean, lib_size_scale)

    # 3) dropout
    if technical_noise_params['add_dropout_effect']:
        binary_ind, prob_ber = dropout_indicator(rng_np, data, technical_noise_params['dropout_shape'], technical_noise_params['dropout_percentile'])
        data = np.multiply(binary_ind, data)
    else:
        prob_ber = None

    # 4) mRNA count data
    if technical_noise_params['return_count_data']:
        dataset_noisy = rng_np.poisson(data)
        dataset_noisy = dataset_noisy.astype(np.float32)
    
    dataset_noisy = jnp.array(dataset_noisy)    
    dataset_noisy.shift = shift
    dataset_noisy.non_drop_prob = prob_ber
    
    return dataset_noisy


""" ---------- SERGIO-type technical noise ---------- """
def dropout_indicator(rng, scData, shape = 1, percentile = 65):
    """
    This is similar to Splat package

    Input:
    scData can be the output of simulator or any refined version of it
    (e.g. with technical noise)

    shape: the shape of the logistic function

    percentile: the mid-point of logistic functions is set to the given percentile
    of the input scData

    returns: np.array containing binary indactors showing dropouts
    """
    scData = np.array(scData)
    scData_log = np.log(np.add(scData,1))
    log_mid_point = np.percentile(scData_log, percentile)
    prob_ber = np.true_divide (1, 1 + np.exp( -1*shape * (scData_log - log_mid_point) ))

    binary_ind = rng.binomial( n = 1, p = prob_ber)

    return binary_ind, prob_ber



""" ---------- data generator ---------- """
 
class MakeDataset():
    def __init__(self, data, data_noisy):
        self.dataset = data
        self.dataset_noisy = data_noisy
        
        if data.mechanism == 'linearGBN':
            self.G = data.g
            self.W = data.W
            self.b = data.b
            self.C = data_noisy.C
        
        if data.mechanism == 'grn_sergio':
            self.G = data.g
            self.cell_type = data.cell_type
        
        if data.mechanism == 'dual':
            self.G = data.g
            self.W = data.W
            self.shift = data_noisy.shift
            self.non_drop_prob = data_noisy.non_drop_prob
            # self.cell_type = data.cell_type
            # self.C = data_noisy.C
    
    def get_original(self):
        return self.dataset
    
    def get_noisy(self):
        return self.dataset_noisy
    


class SyntheticDataGenerator():
    def __init__(self, mechanism, mechanism_params, technical_noise_params, graph_distr, add_technical_noise_params=None, graph=None):
        """
            Generator of synthetic data
            
            Args:
                mechanism (str): which mechanism data is generated from: `linear_additive` or `grn_sergio`
                mechanism_params (dict): dictionary of parameters for the data generating mechanism
                technical_noise_params (dict): dictionary of parameters for additional technical noise to model real data which is often contaminated with some technical measurement error
                graph_distr (Distribution): distribution for sampling a random graph which describes causal relationship among variables in the generated data
                    Example: `functions.synthetic.ErdosRenyi_jax`
        """
        self.mechanism = mechanism
        self.mechanism_params = mechanism_params
        self.technical_noise_params = technical_noise_params
        self.graph_distr = graph_distr
        self.add_technical_noise_params = add_technical_noise_params
        self.graph = graph


    def __call__(self, key, rng_np, num_variables, num_observations, num_observations_int):
        
        start_time_data = time.time()
        
        #---sample graph
        # key, subkey = jax.random.split(key)
        # g = self.graph_distr(subkey, num_variables)
        
        # key, subkey = jax.random.split(key)
        # seed = jax.random.randint(subkey, (1,), 0, 10).item()
        # rng = np.random.default_rng(np.random.SeedSequence(entropy=seed))
        # g = self.graph_distr(rng, num_variables)
        
        key, subkey = jax.random.split(key)

        #---initiate data-generating mechanism and generate data
        if self.mechanism == 'linearGBN':
            model = LinearAdditive(
                        param=self.mechanism_params['weights_distr'],
                        bias=self.mechanism_params['biases_distr'],
                        noise=self.mechanism_params['noise_distr'],
                        noise_scale=self.mechanism_params['noise_scale_distr'],
                        noise_scale_heteroscedastic=None,
                        n_interv_vars=0,
                        interv_dist=None)
            
            # key, subkey = jax.random.split(key)
            # seed = jax.random.randint(subkey, (1,), 0, 10).item()
            # rng = np.random.default_rng(np.random.SeedSequence(entropy=seed))
            
            # seed = jax.random.randint(subkey, (1,), 0, 10).item()
            # rng = np.random.default_rng(np.random.SeedSequence(entropy=seed))
            
            if self.graph is None:
                g = self.graph_distr(rng_np, num_variables)
                # g = self.graph_distr(subkey, num_variables) # TODO: uncomment to test if jax version of graph distr works well (plot prior)
            else:
                g = self.graph
            
            data, W, b = model(rng_np, g, n_observations_obs=num_observations, n_observations_int=num_observations_int)
            
            data = jnp.array(data[0][:, :, 0])
            
            data.mechanism = self.mechanism
            data.g = g
            data.W = W
            data.b = b
            
            key, subkey = jax.random.split(key) 
            data_noisy = apply_technical_noise(subkey, data, self.technical_noise_params) 
            
            
        # if self.mechanism == 'grn_sergio':
        #     model = GRNSergio(**self.mechanism_params, **self.technical_noise_params)
            
        #     if self.graph is None:
        #         # key, subkey = jax.random.split(key)
        #         # g = self.graph_distr(subkey, num_variables)
        #         g = self.graph_distr(rng_np, num_variables)
        #     else:
        #         g = self.graph    
            
        #     key, subkey = jax.random.split(key)
        #     data_noisy_, data_ = model(subkey, g, n_observations_obs=num_observations, n_observations_int=num_observations_int)
            
        #     data_noisy = data_noisy_[0][:, :, 0]
        #     data = data_[:, :, 0]
            
        #     data.mechanism = self.mechanism
        #     data.g = g
        #     data.cell_type = data_.cell_type
            
            
        if self.mechanism == 'grn_sergio':
            model = GRNSergio_np(**self.mechanism_params, **self.technical_noise_params)
            
            if self.graph is None:
                # key, subkey = jax.random.split(key)
                # g = self.graph_distr(subkey, num_variables)
                g = self.graph_distr(rng_np, num_variables)
            else:
                g = self.graph    
            
            key, subkey = jax.random.split(key)
            data_noisy_, data_, cell_types = model(rng_np, g, n_observations_obs=num_observations, n_observations_int=num_observations_int)
            
            data_noisy = jnp.array(data_noisy_[0][:, :, 0])
            data = jnp.array(data_[0][:, :, 0])
            
            data.mechanism = self.mechanism
            data.g = g
            data.cell_type = cell_types
        
        
        if self.mechanism == 'dual':
            model = LinearAdditive(
                        param=self.mechanism_params['weights_distr'],
                        bias=self.mechanism_params['biases_distr'],
                        noise=self.mechanism_params['noise_distr'],
                        noise_scale=self.mechanism_params['noise_scale_distr'],
                        noise_scale_heteroscedastic=None,
                        n_interv_vars=0,
                        interv_dist=None)
            
            if self.graph is None:
                g = self.graph_distr(rng_np, num_variables)
                # g = self.graph_distr(subkey, num_variables) # TODO: uncomment to test if jax version of graph distr works well (plot prior)
            else:
                g = self.graph
            
            data, W, b = model(rng_np, g, n_observations_obs=num_observations, n_observations_int=num_observations_int)
            
            data = jnp.array(data[0][:, :, 0])
            
            data.mechanism = self.mechanism
            data.g = g
            data.W = W
            data.b = b
            
            data_noisy = apply_sergio_technical_noise(rng_np, data, self.technical_noise_params) 
            
        end_time_data = time.time()
        # print(f'time data: {(end_time_data-start_time_data):.2f} sec')
        

        return MakeDataset(data, data_noisy)


""" ---------- data loader (all new datasets) ---------- """

class MetaDataset(Dataset):
    def __init__(self, key, synthetic_data_generator, meta_size, dataset_params):
        self.key = key
        self.data_generator = synthetic_data_generator
        self.meta_size = meta_size
        self.dataset_params = dataset_params
    
    def __len__(self):
        return self.meta_size
    
    def __getitem__(self, idx):
        self.key, subkey = jax.random.split(self.key)
        dataset = self.data_generator(subkey, **self.dataset_params)
        # dataset = self.data_generator(self.key, **self.dataset_params)
        return dataset

def custom_collate(batch):
    """
        batch: output of __getitem__() as defined in MetaDataset class.
               here it is an object of SyntheticDataGenerator() class when it's called as a function with given dataset params (num_obs, num_vars, data preprocessing)
    """
    data_noisy = jnp.stack([item.get_noisy() for item in batch])
    data_true = jnp.stack([item.get_original() for item in batch])
    dataset = [item for item in batch]
    return data_noisy, data_true, dataset


def dataset_loader(key, data_generator, dataset_params, meta_size, batch_size=1):
    meta_dataset = MetaDataset(key, data_generator, meta_size, dataset_params)  
    dataset_loader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    return dataset_loader




#             if self.mechanism == 'dual':
#             model = GRNSergio(**self.mechanism_params, **self.technical_noise_params)
            
#             if self.graph is None:
#                 key, subkey = jax.random.split(key)
#                 g = self.graph_distr(subkey, num_variables)
#             else:
#                 g = self.graph   
            
#             key, subkey = jax.random.split(key)
#             data_noisy_, data_ = model(subkey, g, n_observations_obs=num_observations, n_observations_int=num_observations_int)
            
#             data_noisy_1 = data_noisy_[0][:, :, 0]
#             data_noisy = apply_technical_noise(subkey, data_noisy_1, self.add_technical_noise_params)
#             data = data_[:, :, 0]
            
#             data.mechanism = self.mechanism
#             data.g = g
#             data.cell_type = data_.cell_type
            
#         end_time_data = time.time()
#         # print(f'time data: {(end_time_data-start_time_data):.2f} sec')