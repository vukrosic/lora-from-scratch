"""Step 3. Look at what fine-tuning changed: subtract the old matrix from the new one.

The difference is a matrix too. We ask how complicated it is: in how many directions did the matrix really move?
"""
import numpy as np

import model as M
from step2_full_finetune import get_full

RANKS = [1, 2, 4, 8, 16, 64]


def change(base, tuned):
    """New matrix minus old matrix, for each of the three weight matrices."""
    return {n: tuned[n] - base[n] for n in M.LAYERS}


def strengths(delta):
    """How strongly the change moves along each of its directions, strongest first (the singular values)."""
    return np.linalg.svd(delta.astype(np.float64), compute_uv=False)


def share(s, r):
    """The part of the whole change that the r strongest directions hold."""
    return float((s[:r] ** 2).sum() / (s ** 2).sum())


def run():
    base, tuned = M.get_base(), get_full()[0]
    delta = change(base, tuned)
    out = {}
    print('matrix   size         change / matrix   part of the change in the strongest 1, 2, 4, 8, 16, 64 directions')
    for n in M.LAYERS:
        s = strengths(delta[n])
        size = float(np.linalg.norm(delta[n]) / np.linalg.norm(base[n]))
        sh = [share(s, r) for r in RANKS]
        need = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.9) + 1)
        print(f'  {n}   {delta[n].shape[0]:>4} x {delta[n].shape[1]:<4}   {size:>6.1%}          ' + '  '.join(f'{v:>4.0%}' for v in sh)
              + f'     90% needs {need} of {len(s)}')
        out[n] = {'size': size, 'share': sh, 'need': need, 'of': len(s), 's': s[:64].tolist()}
    return out


if __name__ == '__main__':
    run()
