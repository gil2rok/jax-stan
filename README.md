# Stan + JAX = :heart:

Use your Stan models in JAX via the [BridgeStan](https://github.com/roualdes/bridgestan) library to perform Bayesian inference.

Run multiple chains :chains: in parallel with your favorite sampling algorithm. Never leave JAX again when using Stan.

## Overview

This code provides a minimal example that handles common gotchas and clean design patterns including: 
- BridgeStan model wrapper (`posterior.py`)
- Registering BridgeStan gradients as custom Jacobian-vector-products in JAX -- this is the complex part
- `jit` the sampling step
- `pmap` over multiple cores for parallel sampling, on both CPU and GPU devices (`utils.py`)
- Compiling the Stan model with the `STAN_THREADS=True` flag to enable (CPU) parallelism
- JAX defaults to 32-bit precision while Stan defaults to 64-bit precision

> [!CAUTION]
> Automatic differentiation is performed by Stan on CPUs and then accessed by JAX via BridgeStan. This means **sampling with JAX on GPUs is slow because gradients are still computed on CPUs**. For fast JAX sampling on GPUs consider [NumPyro](https://github.com/pyro-ppl/numpyro) and [Blackjax](https://github.com/blackjax-devs/blackjax) libraries that compute gradients via JAX directly.

See [JAX's tutorial](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#) on custom derivative rules for more information.

## Installation and Setup

1. Follow BridgeStan's instructions to install the prerequisite C++ tool chain [here](https://roualdes.github.io/bridgestan/latest/getting-started.html#getting-started)

2. Clone this repository and install the dependencies in your favorite virtual environment.

    ```bash
    git clone https://github.com/gil2rok/jax-stan.git
    cd jax-stan
    pip install -r requirements.txt
    ```

    Note that in the dependencies (a) `bridgestan` automatically installs the Stan and Stan Math libraries for you and (b) `jax[cuda]` adds optional GPU support with the `[cuda]` flag detailed [here](https://jax.readthedocs.io/en/latest/installation.html).
    
3. Run the example script to sample from the Neal's Funnel model with the No-U-Turn sampler.

    ```bash
    python main.py
    ```
