# thesis: Quick Guide

Folder to look at: **[vae_synthetic4jax](vae_synthetic4jax)** 

## Training pipeline

In the main working directory [`vae_synthetic4jax/`](vae_synthetic4jax/) I run `vae_train_ge.py`. On euler, I run:
```python
python run_euler_ge.py --config_file config_vae2_ge.yaml
```
`config_vae2_ge.yaml` file contains my setup parameters, important ones are under keys *prior*, *VAE_params*, *train*, *slurm*.
`sergio_params_train.yaml` contains parameters for SERGIO simulator.

The whole training loop is within funciton `train()` which is in file `train_ge2.py` in the main working directory [`vae_synthetic4jax/`](vae_synthetic4jax/). The important functions in that file are:
- `train_step()`
- `kl_divergence()`
- `reconstruction()`
- `zero_inflated_poisson_logpmf()`

Other classes or functions called up during training are from subdirectory [`vae_synthetic4jax/functions/`](vae_synthetic4jax/functions/):
- `vae.py`
- `transformer.py`
- `prior.py`

## INSTRUCTIONS: training

### linear GBNs (d=2 \ d=4)
linear Gaussian BN mechanism with our prior, given by the generative model, plus Gaussian correlated noise.

```python
python run_euler.py --config_file config_vae2.yaml
```
- vae_train.py
- train.py
  - uncomment KL with our prior
 
### dual (d=10)
linear Gaussian BN generative mechanism for the signal plus scRNA-seq type of noise on top: Poisson counts and dropout.

```python
python run_euler.py --config_file config_vae2_sergio_dual.yaml
```
- [`vae_train_dual.py`](vae_synthetic4jax/vae_train_dual.py)
- [`train_dual.py`](vae_synthetic4jax/functions/train_dual.py)
- [`data_jax.py`](vae_synthetic4jax/functions/synthetic/data_jax.py)

### linear GBNs + unit Gaussian prior (d=4) [ ablation ]
linear Gaussian BN mechanism with our prior, given by the generative model, plus Gaussian correlated noise.

```python
python run_euler.py --config_file config_vae2_priorTEST.yaml
```
- [`vae_train.py`](vae_synthetic4jax/vae_train.py)
- [`train.py`](vae_synthetic4jax/functions/train.py)
  - uncomment KL with isotropic Gaussian prior

### SERGIO (d=10)
full SERGIO simulator and buffer from AVICI. Possible to add all scRNA-seq noise types.

```python
python run_euler_ge.py --config_file config_vae2_ge.yaml
```
- [`vae_train_ge.py`](vae_synthetic4jax/vae_train_ge.py)
  - buffer calls `sergio_params_train.yml` where noise types are specified
  - buffer calls `SERGIO_noise_config_train.yaml` where we manually set percentile for dropout 
- [`train_ge2.py`](vae_synthetic4jax/functions/train_ge2.py)

## INSTRUCTIONS: testing

### Bayesian causal discovery (DiBS)
We run DiBS on 3 types of data: *noisy*, *learned*, *true*. Here, *learned* is the denoising representation inferred with our trained model.

```python
python run_euler.py --config_file config_dibs_run.yaml
```
In `config_dibs_run.yaml` select one of the following, depending on the test:
- [`dibs_test.py`](vae_synthetic4jax/dibs_test.py) - for standard testing (both linear GBNs and gene expressions (tested SERGIO here too))
- [`dibs_test_DROP2.py`](vae_synthetic4jax/dibs_test_DROP2.py) - central file for running evaluation on different dropout levels and corresponding trained models. Dict sweep_config contains *percentile* key.
- [`dibs_test_linear.py`](vae_synthetic4jax/dibs_test_linear.py) - for testing exclusively linear GBNs with 4 variables (just more convenient as no. of variables and mechanism don't change)

form all these files we can perform both DiBS parameter screen and final evaluation (depending on sweep_config params dictionary).

### Classical causal discovery (PC and GES)
We run PC and GES on 3 types of data: *noisy*, *learned*, *true*. Here, *learned* is the denoising representation inferred with our trained model.

```python
python run_euler_r.py --config_file config_cd_run.yaml
```
In `config_dibs_run.yaml` select one of the following, depending on the test:
- [`cd_methods_test.py`](vae_synthetic4jax/cd_methods_test.py) - for standard testing (both linear GBNs and gene expressions (tested SERGIO here too))
- [`cd_methods_test_linear.py`](vae_synthetic4jax/cd_methods_test_linear.py) - for testing exclusively linear GBNs with 4 variables (just more convenient as no. of variables and mechanism don't change)

form all these files we can perform both DiBS parameter screen and final evaluation (depending on sweep_config params dictionary).

## W&B: results

### training
Project is called **vae_synthetic**.
All runs with linear Gaussian BNs (d=2, d=4), dual (d=10) and SERGIO (d=10).

[`report`](https://wandb.ai/ugne-sak/vae_synthetic/reports/Synthetic-data-d-variables-v1---Vmlldzo1NzI3NjAx)

### evaluation
Project is called **vae_synthetic_dibs**.
Here I've collected all parameter sweeps for tuning DiBS as well as all evaluation results from DiBS, DiBS+ as well as GES and PC.

[`report`](https://wandb.ai/ugne-sak/vae_synthetic/reports/Synthetic-data-d-variables-v1---Vmlldzo1NzI3NjAx](https://wandb.ai/ugne-sak/vae_synthetic_dibs/reports/causal-discovery-tests--Vmlldzo1OTg5NTQw)https://wandb.ai/ugne-sak/vae_synthetic_dibs/reports/causal-discovery-tests--Vmlldzo1OTg5NTQw)



