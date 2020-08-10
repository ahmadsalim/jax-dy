import random
import jax.numpy as np
import jax.random as jaxran
import jax.scipy.stats as sst
import pandas as pd
import numpyro
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from runge_kutta import runge_kutta_4


def seir_model(initial_seir_state, num_days, infection_data, recovery_data, step_size=0.1):
    if infection_data is not None:
        assert num_days == infection_data.shape[0]
    if recovery_data is not None:
        assert num_days == recovery_data.shape[0]
    beta = numpyro.sample('beta', dist.HalfCauchy(scale=1000.))
    gamma = numpyro.sample('gamma', dist.HalfCauchy(scale=1000.))
    sigma = numpyro.sample('sigma', dist.HalfCauchy(scale=1000.))
    mu = numpyro.sample('mu', dist.HalfCauchy(scale=1000.))
    nu = np.array(0.0)  # No vaccine yet

    def seir_update(day, seir_state):
        s = seir_state[..., 0]
        e = seir_state[..., 1]
        i = seir_state[..., 2]
        r = seir_state[..., 3]
        n = s + e + i + r
        s_upd = mu * (n - s) - beta * (s * i / n) - nu * s
        e_upd = beta * (s * i / n) - (mu + sigma) * e
        i_upd = sigma * e - (mu + gamma) * i
        r_upd = gamma * i - mu * r + nu * s
        return np.stack((s_upd, e_upd, i_upd, r_upd), axis=-1)

    num_steps = int(num_days / step_size)
    sim = runge_kutta_4(seir_update, initial_seir_state, step_size, num_steps)
    sim = np.reshape(sim, (num_days, int(1 / step_size), 4))[:, -1, :] + 1e-3
    with numpyro.plate('data', num_days):
        numpyro.sample('infections', dist.Poisson(sim[:, 2]), obs=infection_data)
        numpyro.sample('recovery', dist.Poisson(sim[:, 3]), obs=recovery_data)


if __name__ == "__main__":
    rngkey = jaxran.PRNGKey(random.randint(0, 10000))
    rngkey, rngkey_ = jaxran.split(rngkey)
    num_days = 15
    covid_data = pd.read_csv('data/COVID19_Denmark.csv')
    population = covid_data['Population'][0]
    initial_seir_state = np.array([population - 1., 1., 0., 0.])
    infection_data = covid_data['StillInfected'].to_numpy()
    recovery_data = covid_data['Recovered'].to_numpy() + covid_data[
        'Deaths'].to_numpy()  # Dead people are also recovered in the model
    num_days = infection_data.shape[0]
    step_size = 0.1
    num_warmup = 2500
    num_samples = 10000
    kernel = NUTS(seir_model)
    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains=2)
    mcmc.run(rngkey_, initial_seir_state, num_days, infection_data, recovery_data, step_size=step_size)
    mcmc.print_summary()
