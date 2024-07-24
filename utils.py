# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""Code from https://github.com/pyro-ppl/numpyro/blob/master/numpyro/util.py."""


import os
import re

import jax


def set_platform(platform=None):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    :param str platform: either 'cpu', 'gpu', or 'tpu'.
    """
    if platform is None:
        platform = os.getenv("JAX_PLATFORM_NAME", "cpu")
    jax.config.update("jax_platform_name", platform)


def set_host_device_count(n):
    """
    By default, XLA considers all CPU cores as one device. This utility tells XLA
    that there are `n` host (CPU) devices available to use. As a consequence, this
    allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

    .. note:: This utility only takes effect at the beginning of your program.
        Under the hood, this sets the environment variable
        `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
        `[num_device]` is the desired number of CPU devices `n`.

    .. warning:: Our understanding of the side effects of using the
        `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
        observe some strange phenomenon when using this utility, please let us
        know through our issue or forum page. More information is available in this
        `JAX issue <https://github.com/google/jax/issues/1408>`_.

    :param int n: number of CPU devices to use.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
    )
