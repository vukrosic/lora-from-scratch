"""Quick checks for every step. Run: python3 tests.py  (the first run also trains and saves the base model, about a minute)"""
import os
import tempfile

import numpy as np

import language as L
import model as M
from step4_low_rank import numbers, split, with_change
from step5_lora_layer import LoRA
from step6_train_lora import train_adapter
from step7_merge_swap import merge

# the language: 64 words; every sentence of a style is correct in that style; plain text never uses the style words
assert len(L.VOCAB) == 64
rng = np.random.default_rng(0)
for _ in range(500):
    s = L.plain(rng)
    assert L.style_of(s) == 'plain' and L.style_of(L.pirate(s)) == 'pirate' and L.style_of(L.polite(s)) == 'polite'
    assert not set(s) & set(L.PIRATE + L.POLITE)
assert L.style_of('arr , the sailor finds ye net , matey .'.split()) is None      # "the" is not pirate
assert L.style_of('please , the sailor finds the net , thank you .'.split()) is None   # "kindly" is missing

# the model: 1,644,608 numbers, and a few steps of training already beat blind guessing
fresh = M.new_model(0)
assert M.count(fresh) == 1_644_608
words = L.corpus(300, 'plain', 9)
M.train(fresh, words, 60, 1e-3)
assert M.loss_on(fresh, words) < np.log(64) - 1

# step 4: a column times a row is rebuilt exactly from rank 1, and costs r x (out + in) numbers
table = np.outer([1, 2, 3], [1, 2, 3, 4]).astype(np.float32)
B, A = split(table, 1)
assert B.shape == (3, 1) and A.shape == (1, 4) and np.allclose(B @ A, table, atol=1e-5)
assert numbers(1024, 1024, 4) == 8192

# step 5: a new adapter changes nothing (B is zeros) and holds 18,688 numbers at rank 4
base = M.get_base()
ctx, nxt = M.windows(L.corpus(40, 'pirate', 9))
side = LoRA(base)
assert M.count(side.p) == 18_688
assert np.array_equal(M.forward(base, ctx)[0], M.forward(base, ctx, side)[0])

# step 5: the gradients of A and B agree with nudging one number (on a small model, in exact arithmetic)
M.HID = 16
small = {k: v.astype(np.float64) for k, v in M.new_model(1).items()}
M.HID = 1024
ad = LoRA(small, r=2, seed=1)
ad.p = {k: np.random.default_rng(2).normal(0, 0.3, v.shape) for k, v in ad.p.items()}


def loss():
    return M.backward(small, ctx, M.forward(small, ctx, ad)[1], nxt, ad, base=False)[0]


loss()
grads = {k: v.copy() for k, v in ad.g.items()}
for k in ad.p:
    for i, j in ((0, 0), (1, 1)):
        old = ad.p[k][i, j]
        ad.p[k][i, j] = old + 1e-5
        up = loss()
        ad.p[k][i, j] = old - 1e-5
        down = loss()
        ad.p[k][i, j] = old
        assert abs((up - down) / 2e-5 - grads[k][i, j]) < 1e-6, k

# step 6: training the adapter leaves every number of the base model alone, and lowers the loss
before = M.clone(base)
pirate = L.corpus(300, 'pirate', 9)
side = train_adapter(base, pirate, steps=80)[0]
assert all(np.array_equal(base[k], before[k]) for k in base)
assert M.loss_on(base, pirate, side) < M.loss_on(base, pirate) - 3

# step 7: merging gives the same scores; saving and loading gives the same adapter
a, b = M.forward(base, ctx, side)[0], M.forward(merge(base, side), ctx)[0]
assert np.abs(a - b).max() < 1e-3
path = os.path.join(tempfile.mkdtemp(), 'a.npz')
side.save(path)
again = LoRA.load(base, path)
assert again.scale == side.scale and all(np.array_equal(again.p[k], side.p[k]) for k in side.p)
assert np.array_equal(M.forward(base, ctx, again)[0], a)

# step 4 again: adding B @ A to the matrices is what with_change does
parts = {n: (side.scale * side.p['B' + n[1]], side.p['A' + n[1]]) for n in M.LAYERS}
assert np.abs(M.forward(with_change(base, parts), ctx)[0] - a).max() < 1e-3
print('all tests passed')
