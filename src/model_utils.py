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

    nids, seqs = (tax['ref_rows'], tax['ref_cols'])

    return TaxDescriptor(
        refs,
        ok_pos,
        parents,
        seqs,
        nids,
        paths,
        node_state
    )


def read_lvl(tax_dir):
    tax_dir = Path(tax_dir)
    tax = np.load(tax_dir.resolve()) 
    lvl = tax["node_layer"]
    # getting layer bounds

    bounds = []
    prev = 0
    previ = 0
    for i, v in enumerate(lvl):
        if v != prev:
            bounds.append((previ, i))
            prev = v
            previ = i
    bounds.append((previ, lvl.shape[0]))
    return lvl, bounds


# read_model_jax("", "models/ref_db/taxonomy37k.npz")
print(read_lvl("models/ref_db/taxonomy37k.npz"))