import random
import jax.numpy as np
import jax.random as jaxran
import jax.scipy.stats as sst
import numpyro
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from runge_kutta import runge_kutta_4

def model(obs, step_size=0.2):
    y_0 = numpyro.sample('y_0', dist.HalfNormal(100))
    k = 0.1 + numpyro.sample('k', dist.HalfNormal(100))
    num_steps = int(obs.shape[0] / step_size)
    sim = runge_kutta_4((lambda t, y: -k * y), y_0, step_size, num_steps)
    sim = np.reshape(sim, (obs.shape[0], -1))[..., -1] + 0.1
    with numpyro.plate('data', obs.shape[0], dim=-1):
        numpyro.sample('obs', dist.Poisson(sim), obs=obs)


if __name__ == "__main__":
    true_k = np.array(0.5)
    y_0 = np.array(300.)
    num_days = 50
    num_warmup = 1000
    num_samples = 10000
    x = np.linspace(1, num_days, num=num_days)
    rngkey = jaxran.PRNGKey(random.randint(0, 10000))
    rngkey, rngkey_ = jaxran.split(rngkey)
    obs_rates = y_0 * np.exp(-true_k*x)
    observations = np.maximum(dist.Poisson(obs_rates).sample(rngkey).transpose(), 0)
    step_size = 1.0
    num_steps = int(num_days / step_size)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup, num_samples)
    mcmc.run(rngkey_, observations)
    mcmc.print_summary()