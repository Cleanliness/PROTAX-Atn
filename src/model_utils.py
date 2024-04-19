import jax.numpy as jnp
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

from modules.tree import TaxDescriptor

def read_model_jax(par_dir, tax_dir):
    """
    Read model + tax npz representation
    """
    par_dir = Path(par_dir)
    tax_dir = Path(tax_dir)

    tax = np.load(tax_dir.resolve())
    # par = np.load(par_dir.resolve())
    refs = jnp.array(tax['refs'])
    
    ok_pos = jnp.array(tax['ok_pos'])
    prior = jnp.array(tax['priors'])

    parents = jnp.array(tax['segments'])            # parents of each node
    paths = jnp.array(tax['paths'])
    node_state = jnp.array(tax['node_state'])

    N = parents.shape[0]
    nids, seqs = (tax['ref_rows'], tax['ref_cols'])

    # TODO dumb way of converting to JAX BCSR
    n2s = csr_matrix((np.ones(seqs.shape), (nids, seqs)), shape=(N, refs.shape[0]))

    return TaxDescriptor(
        refs,
        ok_pos,
        parents,
        seqs,
        nids,
        paths,
        node_state
    )


read_model_jax("", "models/ref_db/taxonomy37k.npz")