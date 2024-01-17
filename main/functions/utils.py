import argparse
import torch
import numpy as np
from omegaconf import DictConfig

import functions




def parse_config(config, config_key):
    
    dct = {}
    for key, value in config[config_key].items():
        
        # if isinstance(config[config_key][key], dict):
        # if isinstance(config[config_key][key], DictConfig):
        if isinstance(config[config_key][key], DictConfig) or isinstance(config[config_key][key], dict):
            name = config[config_key][key]['__class__']
            cls = getattr(functions.synthetic, name)
            kwargs = {k: v for k, v in config[config_key][key].items() if k != '__class__'}
            new = cls(**kwargs)
        else:
            new = value
            
        dct[key] = new
    
    return dct


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='config.yaml', help='Name of config file in format [name].yaml which is used for this run')

    return parser


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def get_cov_matrix(correlation_matrix, std):
    """
        inputs:
            correlation_matrix - matrix of correlation coefficients for every variable pair
            std - vector of standard deviations for every variable
        output:
            covariance_matrix - covariance matrix
    """
    assert ((correlation_matrix >= -1) & (correlation_matrix <= 1)).all(), 'some elements in correlation_matrix are not in [-1,1]'
    assert (std >= 0).all(), 'standard deviation is negative'
    
    covariance_matrix = (std.T * std) * correlation_matrix
    
    assert is_pos_def(covariance_matrix), 'covariance matrix is not positive semi-definite'
    
    return covariance_matrix

def get_cov_matrix2(correlation_matrix, std):
    """
        inputs:
            correlation_matrix - matrix of correlation coefficients for every variable pair
            std - vector of standard deviations for every variable
        output:
            covariance_matrix - covariance matrix
    """
    assert ((correlation_matrix >= -1) & (correlation_matrix <= 1)).all(), 'some elements in correlation_matrix are not in [-1,1]'
    assert (std >= 0).all(), 'standard deviation is negative'
    
    std_transp = torch.swapaxes(std, -1, -2)
    covariance_matrix = (std_transp @ std) * correlation_matrix
    
    # assert is_pos_def(covariance_matrix), 'covariance matrix is not positive semi-definite'
    
    return covariance_matrix

