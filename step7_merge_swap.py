"""Step 7. Two things an adapter can do that a fine-tuned copy cannot.

Merge: multiply B by A, add the result into W, and the side path is gone.  Same answers, no extra work per word.
Swap:  keep one base model and one small file per style.  Plug in the one you need.
"""
import os

import numpy as np

import model as M
from step6_train_lora import get_adapter


def merge(base, side):
    """A model with the adapter baked into its matrices: W + (alpha / r) * B @ A."""
    model = M.clone(base)
    for name in M.LAYERS:
        if 'A' + name[1] in side.p:
            model[name] = base[name] + side.scale * side.p['B' + name[1]] @ side.p['A' + name[1]]
    return model


def run():
    base = M.get_base()
    pirate, pirate_path = get_adapter('pirate', base)
    merged = merge(base, pirate)
    ctx = M.windows(M.texts('pirate')[1][:2000])[0]
    a, b = M.forward(base, ctx, pirate)[0], M.forward(merged, ctx)[0]
    gap = float(np.abs(a - b).max())
    agree = float((a.argmax(1) == b.argmax(1)).mean())
    print(f'merged model: {M.count(merged):,} numbers, the same count as the base.  Scores differ by at most {gap:.6f}; '
          f'the top word is the same in {agree:.1%} of {len(ctx):,} places')
    polite, polite_path = get_adapter('polite', base)
    rows = {}
    for label, style, side in (('no adapter', 'plain', None), ('pirate', 'pirate', pirate), ('polite', 'polite', polite)):
        tally = M.styles(M.generate(base, M.start(style), 1000, seed=5, side=side))
        rows[label] = {'tally': tally, 'write': M.write(base, 2, side=side, style=style)}
        print(f'{label:<11} {tally}')
        for s in rows[label]['write']:
            print('     ', s)
    base_bytes = os.path.getsize(os.path.join(M.HERE, 'base.npz'))
    sizes = {'base': base_bytes, 'pirate': os.path.getsize(pirate_path), 'polite': os.path.getsize(polite_path)}
    print(f'files: base.npz {sizes["base"]:,} bytes, pirate.npz {sizes["pirate"]:,}, polite.npz {sizes["polite"]:,}')
    for n in (2, 100):
        print(f'  {n} styles as full copies: {n * base_bytes / 1e6:,.1f} MB.   As one base and {n} adapters: {(base_bytes + n * sizes["pirate"]) / 1e6:,.1f} MB')
    return {'gap': gap, 'agree': agree, 'rows': rows, 'sizes': sizes, 'n_ctx': len(ctx)}


if __name__ == '__main__':
    run()
