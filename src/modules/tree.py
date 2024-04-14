import flax
from flax import struct
import jax.numpy as jnp

@struct.dataclass
class TreeDescriptor:
    num_nodes: int
    num_refs: int
    num_seg: int
    indices: jnp.Array
    segments: jnp.Array