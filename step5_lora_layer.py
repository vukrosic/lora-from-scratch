"""Step 5. The LoRA adapter: next to every weight matrix W, a side path through two thin matrices, A then B.

    output = x @ W.T  +  (alpha / r) * x @ A.T @ B.T

W never changes.  A starts as small random numbers, B starts as zeros, so at the start the side path adds nothing.
"""
import numpy as np

import model as M

RANK = 4


class LoRA:
    def __init__(self, model, r=RANK, alpha=None, seed=0, layers=M.LAYERS):
        rng = np.random.default_rng(seed)
        self.r, self.alpha = r, alpha or 2 * r
        self.scale = self.alpha / r                  # how loud the side path is
        self.p, self.g, self.h = {}, {}, {}          # the numbers we train, their gradients, and what A put out
        for name in layers:
            out, inp = model[name].shape
            self.p['A' + name[1]] = (rng.normal(0, 1, (r, inp)) / np.sqrt(inp)).astype(np.float32)   # r x in, random
            self.p['B' + name[1]] = np.zeros((out, r), np.float32)                                  # out x r, zeros

    def forward(self, name, x):
        """What the side path adds to the output of matrix `name`."""
        if 'A' + name[1] not in self.p:
            return 0
        A, B = self.p['A' + name[1]], self.p['B' + name[1]]
        self.h[name] = x @ A.T                       # squeeze the input down to r numbers
        return self.h[name] @ B.T * self.scale       # and spread them back out

    def backward(self, name, x, d):
        """Gradients for A and B.  d is how the loss changes with the layer's output."""
        if 'A' + name[1] not in self.p:
            return 0
        A, B = self.p['A' + name[1]], self.p['B' + name[1]]
        d = d * self.scale
        self.g['B' + name[1]] = d.T @ self.h[name]
        dh = d @ B
        self.g['A' + name[1]] = dh.T @ x
        return dh @ A                                # what flows on to the layer before

    def save(self, path):
        np.savez(path, alpha=np.float32(self.alpha), **self.p)

    @classmethod
    def load(cls, model, path):
        saved = dict(np.load(path))
        alpha = float(saved.pop('alpha'))
        side = cls(model, r=len(saved[sorted(saved)[0]]), alpha=alpha, layers=['W' + k[1] for k in saved if k[0] == 'A'])
        side.p = saved
        return side


def run():
    base = M.get_base()
    side = LoRA(base)
    total, small = M.count(base), M.count(side.p)
    print(f'rank {side.r} adapter:')
    for k, a in side.p.items():
        print(f'  {k}  {" x ".join(map(str, a.shape)):>9}  {a.size:>6,}')
    print(f'  all           {small:>6,}   next to {total:,} frozen numbers: {small / total:.2%}')
    ctx = M.windows(M.texts('pirate')[1][:500])[0]
    gap = float(np.abs(M.forward(base, ctx)[0] - M.forward(base, ctx, side)[0]).max())
    print(f'B is all zeros, so the model answers exactly as before: biggest difference in any score {gap}')
    return {'total': total, 'small': small, 'shapes': {k: list(a.shape) for k, a in side.p.items()}, 'gap': gap}


if __name__ == '__main__':
    run()
