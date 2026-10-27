"""Single-host sanity check (no jax.distributed): this process should see exactly its host's chips and be able to
run a cross-chip collective on them. Use with scripts/single_host_env.sh."""
import socket

import jax
import jax.numpy as jnp

n = jax.local_device_count()
x = jax.pmap(lambda v: jax.lax.psum(v, "i"), axis_name="i")(jnp.ones(n))
print(
    f"{socket.gethostname()}: process_count={jax.process_count()} devices={jax.device_count()} "
    f"local={n} ids={[d.id for d in jax.local_devices()]} kind={jax.devices()[0].device_kind} psum={float(x[0])}",
    flush=True,
)
