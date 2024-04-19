import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import pandas as pd
import matplotlib.pyplot as plt

from modules.graph_attention import sparse_softmax2_batch
from model_utils import read_model_jax
from models import seqnetShallow

rng = jax.random.PRNGKey(0)

def load_targets(dset_path, split="Train"):
    """
    load target classifications 
    """
    df = pd.read_csv(dset_path)
    arr = df.to_numpy(dtype=int).T

    train_split_idx = int(arr.shape[0] * 0.8)
    val_split_idx = int(arr.shape[0] * 0.9)

    if split == "Train":
        res = arr[:train_split_idx]
    elif split == "Val":
        res = arr[train_split_idx:val_split_idx]
    else:
        res = arr[val_split_idx:]
    
    return jax.random.permutation(rng, res, axis=0)
    

def create_train_state(model, rng, in_shapes, td, hp):
    """
    create train state given
    model: nn.Module
    in_shapes: tuple
    td: TreeDescriptor
    rng: PRNGKey
    hp: hyperparameters dict
    """

    Q = jnp.ones(in_shapes[0], dtype=jnp.uint8)*128
    Q_ok = jnp.ones(in_shapes[1], dtype=jnp.uint8)*128
    params = model.init(rng, Q, Q_ok, td)
    optim = optax.adam(learning_rate=hp['lr'])

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optim
    )


def eval_fn(params, Q, Q_ok, t, td, train_state):
    """
    evaluate loss on batch
    Args:
        params: model parameters
        Q: Query vectors of shape (B, d)
        Q_ok: non gap positions (B, d)
        t: batch targets        (B, N)
        td: Taxonomy descriptor
        train_state: flax train state
    """
    B = Q.shape[0]
    model = train_state.apply_fn
    logits = model(params, Q, Q_ok, td)
    branch = -sparse_softmax2_batch(logits, td.parents, 111813)
    loss = jnp.sum(jnp.take(branch, t))

    return loss

grad_fn = jax.value_and_grad(eval_fn)

@jax.jit
def train_step(Q, Q_ok, t, td, ts):
    """
    One training step over a batched query/training example 
    """
    val, grad = grad_fn(ts.params, Q, Q_ok, t, td, ts)
    # applying gradient update
    updates, new_opt_state = ts.tx.update(grad, ts.opt_state, ts.params)
    new_params = optax.apply_updates(ts.params, updates)
    new_state = ts.replace(params=new_params, opt_state=new_opt_state)

    return new_state, val


def get_batch(td, targets, i):
    Q = td.refs.at[B*i: (i+1)*B].get()
    Q_ok = td.ok_pos.at[B*i: (i+1)*B].get()
    t = jnp.array(targets[B*i: B*(i+1)])

    return Q, Q_ok, t

if __name__ == "__main__":

    train_t = load_targets("data/37k-targets.csv")
    td = read_model_jax("models/ref_db/taxonomy37k.npz")

    # constants
    rng = jax.random.PRNGKey(0)
    B = 5
    d = td.refs.shape[1]*8
    d_ok = td.ok_pos.shape[1]*8
    hp = {
        "lr": 1e-4
    }
    model = seqnetShallow(100, td.N, d, 20)
    ts = create_train_state(model, rng, ((B, td.refs.shape[1]), (B, td.ok_pos.shape[1])), td, hp)

    hist = []

    for i in range(train_t.shape[0] // B):
        Q, Q_ok, t = get_batch(td, train_t, i)
        ts, val = train_step(Q, Q_ok, t, td, ts)
        hist.append(val)

        print(f"minibatch {i}, loss:{val}")

        if i == 500:
            break

    plt.plot(hist)
    plt.savefig("hist.png")