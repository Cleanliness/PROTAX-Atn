import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from modules.graph_attention import sparse_softmax2_batch
from model_utils import read_model_jax
from models import seqnetShallow

rng = jax.random.PRNGKey(0)

def load_targets(dset_path, split="Train"):
    """
    load target classifications. 
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
    

def get_batch(td, targets, i):
    Q = td.refs.at[B*i: (i+1)*B].get()
    Q_ok = td.ok_pos.at[B*i: (i+1)*B].get()
    t = jnp.array(targets[B*i: (i+1)*B])

    return Q, Q_ok, t

# @partial(jax.jit, static_argnums=(0))
def eval_metrics(model, params, td, tset):
    """
    evaluate accuracy on a set of labels
    """

    td_widths = jnp.sum(td.paths != td.paths.shape[0], axis=0)

    matches = 0
    samples = 0

    logp_sum = 0

    for i in range(tset.shape[0] // B):

        # accuracy @ 1
        Q, Q_ok, t = get_batch(td, tset, i)
        logits = model.apply(params, Q, Q_ok, td)
        filled = jnp.take(logits, td.paths, axis=1)
        filled = jnp.nan_to_num(filled, nan=0)
        total = jnp.sum(filled, axis=2)
        chosen = jnp.argmax(total, axis=1)
        chosen_path = jnp.take(td.paths, chosen, axis=0).at[:, 1:].get()

        t_mask = t != -1
        matches += jnp.sum(t_mask*(t == chosen_path), axis=0)
        samples += jnp.sum(t_mask, axis=0)
        logp_sum += (jnp.sum(jnp.sum(filled, axis=1), axis=0) / td_widths).at[1:].get()
        print(f"finish eval on batch {i}")

    return matches / samples, jnp.exp(-logp_sum / samples)

def create_train_state(model, rng, in_shapes, td, hp):
    """
    create train state given
    model: nn.Module
    in_shapes: tuple
    td: TreeDescriptor
    rng: PRNGKey
    hp: hyperparameters dict
    """
    schedule = optax.linear_schedule(hp['lr'], 0.0, 1000)

    Q = jnp.ones(in_shapes[0], dtype=jnp.uint8)*128
    Q_ok = jnp.ones(in_shapes[1], dtype=jnp.uint8)*128
    params = model.init(rng, Q, Q_ok, td)
    optim = optax.adamw(learning_rate=schedule)

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



if __name__ == "__main__":

    train_t = load_targets("data/37k-targets.csv")
    td = read_model_jax("models/ref_db/taxonomy37k.npz")

    # constants
    rng = jax.random.PRNGKey(0)
    B = 5
    d = td.refs.shape[1]*8
    d_ok = td.ok_pos.shape[1]*8
    hp = {
        "lr": 1e-5
    }
    model = seqnetShallow(150, td.N, d, 30)
    ts = create_train_state(model, rng, ((B, td.refs.shape[1]), (B, td.ok_pos.shape[1])), td, hp)

    hist = []
    
    for i in range(train_t.shape[0] // B):
        Q, Q_ok, t = get_batch(td, train_t, i)
        ts, val = train_step(Q, Q_ok, t, td, ts)
        hist.append(val)

        print(f"minibatch {i}, loss:{val}")

        # overfit on a small set
        if i == 200:
            break

    plt.plot(hist)
    plt.xlabel("minibatches")
    plt.ylabel("Cross Entropy")
    plt.savefig("hist.png")

    print("--- acc eval ---")
    acc = eval_metrics(model, ts.params, td, load_targets("data/37k-targets.csv", "Val"))
    print(acc)