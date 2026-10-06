"""Step 2. Fine-tuning the ordinary way: keep training the model on the new text, and let every number change.

It works. Then we look at what it costs.
"""
import os
import time

import numpy as np

import model as M

STEPS, LR = 1000, 3e-4


def finetune_all(base, words, steps=STEPS, lr=LR, log=None):
    """A copy of the model, trained further on `words`. Returns the copy and the optimizer."""
    tuned = M.clone(base)
    return tuned, M.train(tuned, words, steps, lr, log=log)


def get_full(style='pirate'):
    """The fully fine-tuned copy, saved as its own file: full_pirate.npz."""
    path = os.path.join(M.HERE, f'full_{style}.npz')
    if not os.path.exists(path):
        np.savez(path, **finetune_all(M.get_base(), M.texts(style)[0])[0])
    return dict(np.load(path)), path


def run():
    base = M.get_base()
    pirate, pirate_test = M.texts('pirate')
    plain_test = M.texts('plain')[1]
    print(f'pirate text: 3,000 sentences, {len(pirate):,} words.  For example:')
    print('  ', ' '.join(M.VOCAB[i] for i in pirate[:22]))
    log, t = [], time.time()
    tuned, opt = finetune_all(base, pirate, log=log)
    secs = time.time() - t
    get_full()
    path = os.path.join(M.HERE, 'full_pirate.npz')
    before, after = M.loss_on(base, pirate_test), M.loss_on(tuned, pirate_test)
    print(f'{STEPS:,} steps, every number allowed to change: {secs:.0f} s')
    print(f'loss on pirate text: {before:.2f} -> {after:.3f}')
    for s in M.write(tuned, 4, style='pirate'):
        print('  ', s)
    tally = M.styles(M.generate(tuned, M.start('pirate'), 1000, seed=5))
    print(f'1,000 sentences written: {tally["pirate"]} correct pirate sentences')
    total = M.count(base)
    print('what it cost:')
    print(f'  numbers trained       {total:>10,}  (100% of the model)')
    print(f'  optimizer memory      {opt.memory():>10,}  numbers (two for every number trained)')
    print(f'  file for this style   {os.path.getsize(path):>10,}  bytes, a whole second model')
    plain_before, plain_after = M.loss_on(base, plain_test), M.loss_on(tuned, plain_test)
    forgot = M.styles(M.generate(tuned, M.start(), 1000, seed=6))
    print(f'  and the copy forgot:  loss on plain text {plain_before:.3f} -> {plain_after:.2f}; it writes {forgot["plain"]} plain sentences in 1,000')
    return {'log': log, 'secs': secs, 'before': before, 'after': after, 'tally': tally, 'forgot': forgot['plain'], 'total': total,
            'opt': opt.memory(), 'bytes': os.path.getsize(path), 'plain_before': plain_before, 'plain_after': plain_after}


if __name__ == '__main__':
    run()
