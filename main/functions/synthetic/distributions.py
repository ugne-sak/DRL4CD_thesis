from functions.synthetic.abstract import Distribution


class Gaussian(Distribution):
    def __init__(self, scale=1.0, mean=0.0):
        self.mean = mean
        self.scale = scale

    def __call__(self, rng, shape=None):
        return self.mean + self.scale * rng.normal(size=shape)


class Laplace(Distribution):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, rng, shape=None):
        return self.scale * rng.laplace(size=shape)


class Cauchy(Distribution):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, rng, shape=None):
        return self.scale * rng.standard_cauchy(size=shape)


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, rng, shape=None):
        return rng.uniform(size=shape, low=self.low, high=self.high)


class SignedUniform(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, rng, shape=None):
        sgn = rng.choice([-1, 1], size=shape)
        return sgn * rng.uniform(size=shape, low=self.low, high=self.high)


class RandInt(Distribution):
    def __init__(self, low, high, endpoint=True):
        self.low = low
        self.high = high
        self.endpoint = endpoint

    def __call__(self, rng, shape=None):
        return rng.integers(size=shape, low=self.low, high=self.high, endpoint=self.endpoint)


class Beta(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, rng, shape=None):
        return rng.beta(self.a, self.b, size=shape)



#-------
import jax
import jax.numpy as jnp

class Gaussian_jax(Distribution):
    def __init__(self, scale=1.0, mean=0.0):
        self.mean = mean
        self.scale = scale

    def __call__(self, rng, shape=None):
        return self.mean + self.scale * jax.random.normal(rng, shape=shape)

class Uniform_jax(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, rng, shape=None):
        return jax.random.uniform(rng, shape=shape, minval=self.low, maxval=self.high)

class SignedUniform_jax(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, rng, shape=None):
        sgn = jax.random.choice(rng, jnp.array([-1, 1]), shape=shape)
        return sgn * jax.random.uniform(rng, shape=shape, minval=self.low, maxval=self.high)

class Beta_jax(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, rng, shape=None):
        return jax.random.beta(rng, a=self.a, b=self.b, shape=shape)

class RandInt_jax(Distribution):
    def __init__(self, low, high, endpoint=True):
        self.low = low
        self.high = high
        self.endpoint = endpoint

    def __call__(self, rng, shape=None):
        return jax.random.randint(rng, shape=shape, minval=self.low, maxval=self.high + 1 if self.endpoint else self.high)