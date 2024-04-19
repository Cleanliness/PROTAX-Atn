import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from model_utils import read_model_jax
from modules.graph_attention import *

class seqnetShallow(nn.Module):
    """
    'Shallow' model intended to match baseline PROTAX 
    but replaces KNN with cross attention
    """
    hid_dim: int    # attention layer output dim
    out_dim: int    # logits to output
    seq_dim: int    # dimensions of one sequence
    ne_dim: int     # node embedding dimension

    def setup(self):
        # self.W_final = self.param('W_final', nn.initializers.lecun_normal(), (1, self.hid_dim))     # also includes binary state
        self.b_o = self.param('b_o', nn.initializers.zeros, (1, self.out_dim)) 
        self.node_embed = self.param('state_embd', nn.initializers.lecun_normal(), (2, self.ne_dim))
        self.W_res = self.param('W_res', nn.initializers.lecun_normal(), (self.seq_dim, self.hid_dim))
        self.atn = seqsim_atn(self.hid_dim, self.seq_dim)

    def __call__(self, Q, Q_ok, td):
        """
        perform batch FP

        Args:
            Q - Query vectors of shape (B, dk)
            Q_ok - non gap positions (B, dk)
            td - Descriptor of sparsity structure (defines atn neighborhoods for each (V)alue)
        """
        Q_unpacked = jnp.unpackbits(Q, axis=1)
        Q_proj = jnp.expand_dims(Q_unpacked @ self.W_res, axis=1)
        reweighted_nodes = jnp.sum(self.atn(Q, Q_ok, td)*Q_proj, axis=2)     # (B, N, W_o) -> (B, N)
        node_embeds = jnp.expand_dims(jnp.sum(td.node_state@self.node_embed, axis=1), 0)
        return reweighted_nodes + node_embeds + self.b_o


