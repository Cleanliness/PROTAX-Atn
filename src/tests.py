import jax.numpy as jnp
from model_utils import read_model_jax
from modules.graph_attention import *
from models import seqnetShallow

td = read_model_jax("models/ref_db/taxonomy37k.npz")

# some constants for testing
B = 5
rng = jax.random.PRNGKey(0)
q = td.refs[0]
q_ok = td.ok_pos[0]
dists = seqsim(q, q_ok, td)
num_seg = td.N

q_batch = td.refs[:B]
q_ok_batch = td.ok_pos[:B]

# ========== Tests ==========
def test_softmax():
    applied = jnp.take(dists, td.ref2seg)
    s = sparse_softmax(applied, td.segments, num_seg)
    s2 = jnp.exp(sparse_softmax2(applied, td.segments, num_seg))

    assert jnp.sum((s-s2)*(s-s2)) < 1e-8
    print("softmax correctness passed")


def test_QKV_seqdist():
    V = QKV_seqdist(q, q_ok, td)
    assert V.shape[0] == td.N
    print("QKV correctness passed")


def test_QKV_seqdist_batched():
    V = QKV_seqdist_batched(q_batch, q_ok_batch, td)
    assert V.shape[:-1] == (B, td.N)

    print("QKV batched correctness passed")


def test_seqdist_atn():

    net = seqsim_atn(10, td.refs.shape[1]*8)
    params = net.init(rng, q_batch, q_ok_batch, td)
    
    res = net.apply(params, q_batch, q_ok_batch, td)

    print("seqdist atn passed", res.shape)


def test_seqnet():
    net = seqnetShallow(10, td.N, td.refs.shape[1]*8, 5)
    params = net.init(rng, q_batch, q_ok_batch, td)
    res = net.apply(params, q_batch, q_ok_batch, td)
    print("seqnet passed")


if __name__ == "__main__":
    test_softmax()
    test_QKV_seqdist()
    test_QKV_seqdist_batched()
    test_seqdist_atn()
    test_seqnet()
    print("tests done")


