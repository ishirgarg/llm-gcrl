"""Multi-host sanity check: every worker joins, sees all chips, and a cross-host psum works."""
import jax
import jax.numpy as jnp

jax.distributed.initialize()
n = jax.device_count()
x = jax.pmap(lambda v: jax.lax.psum(v, "i"), axis_name="i")(jnp.ones(jax.local_device_count()))
print(
    f"process {jax.process_index()}/{jax.process_count()}: "
    f"{jax.local_device_count()} local / {n} global devices ({jax.devices()[0].device_kind}); "
    f"psum={float(x.addressable_shards[0].data[0])} (expect {n})",
    flush=True,
)
