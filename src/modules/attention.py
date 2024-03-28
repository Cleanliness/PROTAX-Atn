# local dot product attention
import jax
import jax.numpy as jnp

# number of segments
NUM_SEGMENTS = 4

def local_attention(sim, nodes, seg, refs, V):
    """
    local attention over nodes assigned to
    a sequence
    """
    
    # assigning similarities to correct nodes
    nodes_assigned = jnp.take(dists, nodes)
    max_z = jnp.take(jax.ops.segment_max(nodes_assigned, seg, num_segments=NUM_SEGMENTS), seg)
    nodes_assigned -= max_z
    nodes_exp = jnp.exp(nodes_assigned)
    segsum = jax.ops.segment_sum(nodes_assigned, seg, num_segments=NUM_SEGMENTS), seg)
    nodes_sum = jnp.take(segsum, seg)

    atn_scores = nodes_assigned - jnp.log(segsum)
    atn_scores = jnp.exp(atn_scores)

    # reweighting value matrix (i.e. sequences)
    return V @ atn_scores
    
