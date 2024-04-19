import flax
from flax import struct
import jax.numpy as jnp

@struct.dataclass
class TreeDescriptor:
    num_nodes: int
    num_edges: int          # i.e. number of nonzero entries in sparse rep
    num_seg: int            # number of neighborhoods to attend over
    node2id: jnp.array      # map segment id to nodes
    segment_id: jnp.array     # neighborhood definition


@struct.dataclass
class TaxDescriptor:
    """
    Taxonomic tree descriptor

    N = total number of nodes
    L = Number of non-species nodes (i.e nodes with depth < 7)
    R = total number of reference sequences
    p = path length, i.e. depth of taxonomy
    E = number of edges in node to sequence bipartite graph

    refs: All reference sequences   (i.e. queries and values for seqdist atn)
    ok_pos: positions which contain a, t, c, g
    parents: parent index for each node
    ref2seg: maps each reference sequence to position in neighborhood
    segments: neighborhood of reference seqs for each node
    paths: path to each node from root
    node_state: binary variables of node state (unknown, num_refs)
    """
    refs: jnp.array                  # [R, dr]
    ok_pos: jnp.array                # [R]
    parents: jnp.array               # [N] (sorted)
    ref2seg: jnp.array              # [E] 
    segments: jnp.array              # [E]
    paths: jnp.array                 # [N, p]
    node_state: jnp.array            # [N, 2]

    @property
    def N(self):
        return self.parents.shape[0]
    
    @property
    def R(self):
        return self.refs.shape
