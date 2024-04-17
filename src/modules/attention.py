# local dot product attention
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

# my imports
import tree
import jraph

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
    

class GraphAttentionLayer(nn.Module):
    out_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    def setup(self):
        # Linear transformations for each head
        self.proj_query = nn.DenseGeneral(features=(self.num_heads, self.out_dim))
        self.proj_key = nn.DenseGeneral(features=(self.num_heads, self.out_dim))
        self.proj_value = nn.DenseGeneral(features=(self.num_heads, self.out_dim))
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, graph, deterministic: bool = True):
        # Node features
        node_feats = graph.nodes

        # Linear projections
        Q = self.proj_query(node_feats)
        K = self.proj_key(node_feats)
        V = self.proj_value(node_feats)

        # Scaled dot-product attention
        attention_scores = jnp.einsum('bhd,bid->bhi', Q, K) / jnp.sqrt(self.out_dim)
        attention_scores = jax.nn.softmax(attention_scores, axis=-1)
        
        # Optional dropout for training
        attention_scores = self.dropout(attention_scores, deterministic=deterministic)

        # Weighted sum of values
        out = jnp.einsum('bhi,bhd->bid', attention_scores, V)

        # Combine heads
        out = jnp.reshape(out, (out.shape[0], -1))

        return out

# # Example usage
# num_nodes = 10
# num_features = 16
# num_heads = 4
# out_dim = 8

# nodes = jnp.ones((num_nodes, num_features))  # Dummy node features
# graph = jraph.GraphsTuple(
#     nodes=nodes,
#     edges=None,
#     senders=None,
#     receivers=None,
#     globals=None,
#     n_node=jnp.array([num_nodes]),
#     n_edge=jnp.array([0])
# )

# gat_layer = GraphAttentionLayer(out_dim=out_dim, num_heads=num_heads)
# out_features = gat_layer(graph)
# print(out_features.shape)