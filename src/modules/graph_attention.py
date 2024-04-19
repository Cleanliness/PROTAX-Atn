# local dot product attention
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

# my imports
from tree import TreeDescriptor

class graph_atn(nn.Module):
    """
    general graph attention 
    works with sparse representation defining neighborhoods
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
            K - Key vectors of shape (R, dk)
            td - Descriptor of sparsity structure
        """
        B, R, dk = K.shape
        QK = X @ K / dk        
        scores = sparse_softmax(QK, td.inds, td.seg, td.num_seg)


class seg_atn(nn.Module):
    """
    self attention attending over sparse structure.
    specialized for bipartite graphs with contiguous neighborhoods
    """
    out_dim: int
    dk: int

    def setup(self):
        self.W_o = nn.Dense(self.out_dim)

    def __call__(Q, K, td):
        """
        perform batch forward pass of attention
        Args:
            Q - Query vectors of shape (B, dk)
            K - Key vectors of shape (R, dk)
            td - Descriptor of sparsity structure (defines atn neighborhoods for each (V)alue)
        """

        QK = Q @ K.T


def sparse_softmax(X, inds, seg, num_seg):
    """
    Performs softmax over segments

    Args:
        X       - input to take softmax over (Num_x)
        inds    - indices of shape (num_x)
        seg     - indices corresponding to neighborhoods 
                    where softmax is applied
        num_seg - number of segments (for JIT compilation)
    """
    
    # logsumexp for numerical stability
    applied = jnp.take(inds, X)
    max_x = jnp.ops.segment_max()
    

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    # 2 keys
    Q = jnp.ones((2, 4))
    K = jax.random.uniform(rng, (6, 4))

    # neighborhood
    td = TreeDescriptor(
        num_nodes=3,
        num_edges=7,
        num_seg=5,
        indices = jnp.array([])
    )
