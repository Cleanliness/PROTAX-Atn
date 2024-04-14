# local dot product attention
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

# my imports
import tree

class sparse_atn(nn.Module):
    """
    Self attention attending sparse structure
    Can correspond to tree or more abstract
    """
    out_dim: int    # output dimensionality
    dk: int         # key dim

    def setup(self):
        self.W_o = nn.Dense(self.out_dim)

    def __call__(X, K, td):
        """
        performs batch pass of attention on a sparse matrix
        Assumes batch operates on same graph

        Args:
            X - Query vectors of shape (B, dk)
            K - Key vectors of shape (B, R, dk)
            td - Tree descriptor describing sparsity structure
        """
        B, R, dk = K.shape
        QK = X @ K / dk        
        scores = sparse_softmax(QK, td.inds, td.seg, td.num_seg)


def sparse_softmax(X, inds, seg, num_seg):
    """
    Performs softmax over segments

    Args:
        X - input of shape (Batch, Num_x)
        inds    - indices of shape (num_x)
        seg     - indices corresponding to neighborhoods 
                    where softmax is applied
        num_seg - number of segments (for JIT compilation)
    """
    
    # logsumexp for numerical stability
    applied = jnp.take(inds, X)
    max_x = jnp.ops.segment_max()
    


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
    
