"""Can a multi-controller JAX process run a collective over ONLY some of its own chips (as a per-host vLLM replica
would)? Tries every subset of size >= 2 (in id order). Run on all workers at once (multi-host mode)."""
import itertools

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

jax.distributed.initialize()
local = sorted(jax.local_devices(), key=lambda d: d.id)
ok, bad = [], []
for k in range(2, len(local) + 1):
  for devs in itertools.combinations(local, k):
    mesh = Mesh(np.array(devs), ("model",))
    x = jax.device_put(jnp.ones((len(devs), 8), jnp.float32), NamedSharding(mesh, P("model")))
    try:
      jax.block_until_ready(jax.jit(lambda a: a.sum(axis=0), out_shardings=NamedSharding(mesh, P()))(x))
      ok.append([d.id for d in devs])
    except Exception:  # pylint: disable=broad-except
      bad.append([d.id for d in devs])
print(f"process {jax.process_index()} ids={[d.id for d in local]}: OK={ok} FAIL={bad}", flush=True)
