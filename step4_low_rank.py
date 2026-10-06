"""Step 4. A cheap way to write down a simple change: two thin matrices instead of one big one.

A table that is "one column times one row" needs only the column and the row.
r columns times r rows gives a table of rank r.  That is the whole trick of LoRA.
"""
import numpy as np

import model as M
from step2_full_finetune import get_full
from step3_the_change import RANKS, change


def numbers(out, inp, r):
    """How many numbers the two thin matrices hold: B is out x r, A is r x inp."""
    return r * (out + inp)


def split(delta, r):
    """The best rank-r copy of a matrix, as two thin matrices B (out x r) and A (r x in)."""
    U, s, Vt = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    return (U[:, :r] * s[:r]).astype(np.float32), Vt[:r].astype(np.float32)


def with_change(base, parts):
    """The base model with B @ A added to each matrix."""
    model = M.clone(base)
    for n, (B, A) in parts.items():
        model[n] = base[n] + B @ A
    return model


def run():
    column, row = np.array([[1], [2], [3]]), np.array([[1, 2, 3, 4]])
    print('one column times one row:')
    print(column @ row)
    print(f'the table holds {column.size * row.size} numbers; the column and the row hold {column.size + row.size}')
    out, inp = M.HID, M.HID
    print(f'W2 is {out} x {inp} = {out * inp:,} numbers.  A change of rank r costs r x ({out} + {inp}):')
    for r in RANKS:
        print(f'  r = {r:<3} {numbers(out, inp, r):>8,}   {numbers(out, inp, r) / (out * inp):.2%}')
    base, tuned = M.get_base(), get_full()[0]
    delta, test = change(base, tuned), M.texts('pirate')[1]
    rows = []
    print('keep only the r strongest directions of the fine-tuning change (everything else stays as in the base):')
    for r in RANKS + [None]:
        parts = {n: split(delta[n], r or min(delta[n].shape)) for n in M.LAYERS}
        model = with_change(base, parts)
        loss, ok = M.loss_on(model, test), M.styles(M.generate(model, M.start('pirate'), 1000, seed=5))['pirate']
        rows.append({'r': r, 'loss': loss, 'ok': ok})
        print(f'  r = {str(r or "all"):<4} loss on pirate text {loss:>6.3f}   correct pirate sentences {ok:>4} of 1,000')
    return {'rows': rows, 'table': (column @ row).tolist()}


if __name__ == '__main__':
    run()
