# local dot product attention
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

# my imports

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


class seqsim_atn(nn.Module):
    """
    sparse cross attention utilizing sequence similarity
    specialized for bipartite graphs 
    assumes contiguous neighborhoods in Keys
    """
    out_dim: int
    d: int              # sequence dimensionality

    def setup(self):
        self.W_o = self.param('W_o', nn.initializers.lecun_normal(), (self.d, self.out_dim))
        self.b_o = self.param('b_o', nn.initializers.zeros, (self.out_dim,))

    def __call__(self, Q, Q_ok, td):
        """
        perform batch forward pass of attention
        Args:
            Q - Query vectors of shape (B, dk)
            Q_ok - non gap positions (B, dk)
            td - Descriptor of sparsity structure (defines atn neighborhoods for each (V)alue)
        """

        V = QKV_seqdist_batched(Q, Q_ok, td)
        z = V @ jnp.expand_dims(self.W_o, 0) + self.b_o    # (B, N, d) -> (B, N, d_o)
        return z     

def seqsim(q, ok_query, td):
    """
    pairwise sequence similarity, normalized by length
    i.e. # matches / len

    Args:
        q - query sequence (d,)
        td - tree descriptor
        ok_query - non gap positions in query sequence
    """

    # count matches and valid positions
    ok = jnp.bitwise_and(ok_query, td.ok_pos)
    ok = jnp.sum(jax.lax.population_count(ok), axis=1)
    match = jnp.bitwise_and(q, td.refs)

    match_tots = jnp.sum(jax.lax.population_count(match), axis=1)
    return match_tots / ok

seqsim_batched = jax.vmap(seqsim, (0, 0, None), 0)


def sparse_softmax2(X, inds, seg, num_seg):
    """
    Numerically stable softmax for training

    Args:
        X       - input to take softmax over (d,)
        inds    - maps input to segment dim (s,) max=d-1
        seg     - indices corresponding to neighborhoods  (s)
        num_seg - number of segments (for JIT compilation)
    """

    # logsumexp applied over all segments
    applied = jnp.take(X, inds)
    max_terms = jnp.take(jax.ops.segment_max(applied, seg, num_segments=num_seg), seg)
    applied -= max_terms
    norm_terms = jnp.take(jnp.log(jax.ops.segment_sum(jnp.exp(applied), seg, num_segments=num_seg)), seg)
    return jnp.exp(applied - norm_terms)


def sparse_softmax(X, inds, seg, num_seg):
    """
    Performs softmax over segments for inference

    Args:
        X       - input to take softmax over (d,)
        inds    - maps input to segment dim (s,) max=d-1
        seg     - indices corresponding to neighborhoods  (s)
        num_seg - number of segments (for JIT compilation)
    """
    
    applied = jnp.take(jnp.exp(X), inds)
    norm_terms = jnp.take(jax.ops.segment_sum(applied, seg, num_segments=num_seg), seg)
    return applied / norm_terms

    # max_x = jnp.ops.segment_max()


def QKV(q, ok_query, td):
    dists = seqsim(q, ok_query, td)
    scores = sparse_softmax2(dists, td.ref2seg, td.segments, td.N)
    applied_values = jnp.take(td.refs, td.ref2seg)*scores
    return jax.ops.segment_sum(applied_values, td.segments, num_segments=td.N)

def QKV_seqdist(q, ok_query, td):
    dists = seqsim(q, ok_query, td)
    scores = sparse_softmax2(dists, td.ref2seg, td.segments, td.N)
    values = jnp.unpackbits(td.refs, axis=1)
    applied_values = jnp.take(values, td.ref2seg, axis=0)
    applied_values *= jnp.expand_dims(scores, 1)
    return jax.ops.segment_sum(applied_values, td.segments, num_segments=td.N)

QKV_seqdist_batched = jax.jit(jax.vmap(QKV_seqdist, (0, 0, None), 0))

if __name__ == "__main__":
    pass
    
