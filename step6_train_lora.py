"""Step 6. Train only the adapter.  The model's own numbers are frozen: no gradient is kept for them, nothing updates them.

Then compare with full fine-tuning from step 2: quality, numbers trained, memory, file size.
"""
import os
import time

import numpy as np

import model as M
from step2_full_finetune import finetune_all
from step5_lora_layer import RANK, LoRA

STEPS, LR = 1000, 3e-3


def train_adapter(base, words, r=RANK, alpha=None, steps=STEPS, lr=LR, log=None, layers=M.LAYERS):
    side = LoRA(base, r, alpha, layers=layers)
    return side, M.train(base, words, steps, lr, side=side, log=log)


def get_adapter(style, base=None):
    """The rank-4 adapter for one style, saved as its own small file: pirate.npz or polite.npz."""
    base = base or M.get_base()
    path = os.path.join(M.HERE, f'{style}.npz')
    if not os.path.exists(path):
        train_adapter(base, M.texts(style)[0])[0].save(path)
    return LoRA.load(base, path), path


def run():
    base = M.get_base()
    frozen = M.clone(base)
    pirate, test = M.texts('pirate')
    plain_test = M.texts('plain')[1]
    log, t = [], time.time()
    side, opt = train_adapter(base, pirate, log=log)
    secs = time.time() - t
    path = get_adapter('pirate', base)[1]
    same = all(np.array_equal(base[k], frozen[k]) for k in base)
    loss = M.loss_on(base, test, side)
    tally = M.styles(M.generate(base, M.start('pirate'), 1000, seed=5, side=side))
    print(f'{STEPS:,} steps on the adapter only: {secs:.0f} s.  Base model untouched: {same}')
    for s in M.write(base, 4, side=side, style='pirate'):
        print('  ', s)
    full_log, t = [], time.time()
    full, full_opt = finetune_all(base, pirate, log=full_log)
    full_secs = time.time() - t
    full_loss = M.loss_on(full, test)
    full_tally = M.styles(M.generate(full, M.start('pirate'), 1000, seed=5))
    total, small = M.count(base), M.count(side.p)
    full_bytes = os.path.getsize(os.path.join(M.HERE, 'full_pirate.npz'))
    print('                              full fine-tune      LoRA rank 4')
    print(f'  numbers trained             {total:>12,}   {small:>12,}   ({small / total:.2%})')
    print(f'  loss on pirate text         {full_loss:>12.3f}   {loss:>12.3f}')
    print(f'  correct sentences of 1,000  {full_tally["pirate"]:>12}   {tally["pirate"]:>12}')
    print(f'  optimizer memory (numbers)  {full_opt.memory():>12,}   {opt.memory():>12,}')
    print(f'  file for the style (bytes)  {full_bytes:>12,}   {os.path.getsize(path):>12,}')
    print(f'  training time               {full_secs:>10.1f} s   {secs:>10.1f} s')
    print(f'  plain text after (loss)     {M.loss_on(full, plain_test):>12.2f}   {M.loss_on(base, plain_test):>12.3f}   (take the adapter off and the base is still there)')
    return {'log': log, 'full_log': full_log, 'secs': secs, 'full_secs': full_secs, 'loss': loss, 'full_loss': full_loss,
            'ok': tally['pirate'], 'full_ok': full_tally['pirate'], 'opt': opt.memory(), 'full_opt': full_opt.memory(),
            'bytes': os.path.getsize(path), 'full_bytes': full_bytes, 'small': small, 'total': total, 'same': same,
            'plain_full': M.loss_on(full, plain_test), 'plain_base': M.loss_on(base, plain_test)}


if __name__ == '__main__':
    run()
