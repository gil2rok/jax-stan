import multiprocessing

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import utils
from posterior import Posterior


def make_logdensity_fn(posterior):
    """Register a Stan model with JAX's custom VJP system via Bridgestan.

    See https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html.
    """

    @jax.custom_vjp
    def logdensity_fn(arg):
        # Cast to float64 to match Stan's dtype
        fn = lambda x: posterior.logdensity(np.array(x, dtype=np.float64))
        # Cast back to float32 to match JAX's default dtype
        result_shape = jax.ShapeDtypeStruct((), jnp.float32)
        return jax.pure_callback(fn, result_shape, arg)

    def call_grad(arg):
        fn = lambda x: posterior.logdensity_and_gradient(np.array(x, dtype=np.float64))[
            1
        ]
        result_shape = jax.ShapeDtypeStruct(arg.shape, arg.dtype)
        return jax.pure_callback(fn, result_shape, arg)

    def vjp_fwd(arg):
        return logdensity_fn(arg), arg

    def vjp_bwd(residuals, y_bar):
        arg = residuals
        return (call_grad(arg) * y_bar,)

    logdensity_fn.defvjp(vjp_fwd, vjp_bwd)

    return logdensity_fn


def inference_loop(rng_key, kernel, initial_state, num_samples):
    """Run a parallelizable inference loop when sampling with JAX.

    See https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html
    """

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state.position

    keys = random.split(rng_key, num_samples)
    final_state, positions = jax.lax.scan(one_step, initial_state, keys)

    return positions


def main():
    seed = 0
    rng_key = random.key(seed)
    num_chains = jax.device_count()

    posterior = Posterior(seed=seed, model_path="funnel10.stan", data_path=None)
    logdensity_fn = make_logdensity_fn(posterior)

    step_size = 1e-3
    inverse_mass_matrix = jnp.ones(posterior.dims)
    nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

    initial_positions = jnp.ones((num_chains, posterior.dims))
    initial_states = jax.pmap(nuts.init, in_axes=(0))(initial_positions)

    num_samples = 1000
    rng_key, sample_key = random.split(rng_key)
    sample_keys = random.split(sample_key, num_chains)

    inference_loop_pmap = jax.pmap(
        inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3)
    )
    positions = inference_loop_pmap(sample_keys, nuts.step, initial_states, num_samples)
    print(positions.shape)


if __name__ == "__main__":
    # WARNING: Using Stan's automatic differentiation (via the Bridgestan package)
    # on GPUs is very slow because Stan computes all gradients on the CPU.

    ####### Run on CPU #######
    utils.set_platform("cpu")
    utils.set_host_device_count(multiprocessing.cpu_count())
    print(f"Running on {jax.device_count()} CPU cores.")

    ####### Run on GPU #######
    # utils.set_platform("gpu")
    # print(f"Running on {jax.device_count()} GPU cores.")

    main()
