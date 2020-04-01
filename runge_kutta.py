import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
from jax import ops
from jax.experimental import loops


@functools.partial(jax.jit, static_argnums=(0,))
def runge_kutta_4(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                  y0: np.ndarray,
                  step_size=0.01,
                  num_steps=1000):
    with loops.Scope() as s:
        s.res = np.empty(num_steps)
        s.y = y0
        for i in s.range(num_steps):
            t = i * step_size
            k1 = step_size * f(t, s.y)
            k2 = step_size * f(t + step_size / 2, s.y + k1 / 2)
            k3 = step_size * f(t + step_size / 2, s.y + k2 / 2)
            k4 = step_size * f(t + step_size, s.y + k3)
            s.y = s.y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            s.res = ops.index_update(s.res, i, s.y)
        return s.res

if __name__ == "__main__":
    rkap = runge_kutta_4(lambda t, y: - 2 * y , np.array(3.), step_size=0.05)
    ex = np.array([3*np.exp(-2*(i+1)*0.05) for i in range(1000)])
    print(np.max(rkap - ex))
