"""Step 1. The base model: train it on the plain language, count its numbers, let it write.

This is the model we will later teach a new style. Everything after this step starts from it.
"""
import numpy as np

import model as M


def run():
    log = []
    base = M.get_base(log)
    train_text, test_text = M.texts('plain')
    print(f'trained on {len(train_text):,} words of plain text, 5,000 steps')
    print('the numbers inside the model:')
    for name, a in base.items():
        print(f'  {name:<3} {" x ".join(map(str, a.shape)):>11}  {a.size:>9,}')
    total = M.count(base)
    in_w = sum(base[n].size for n in M.LAYERS)
    print(f'  all            {total:>9,}   ({in_w / total:.1%} of them sit in W1, W2 and W3)')
    loss = M.loss_on(base, test_text)
    print(f'loss on plain text it never saw: {loss:.3f}   (guessing blindly among 64 words: {np.log(64):.3f})')
    for s in M.write(base, 4):
        print('  ', s)
    tally = M.styles(M.generate(base, M.start(), 1000, seed=5))
    print(f'1,000 sentences written: {tally["plain"]} correct plain sentences, {tally["pirate"]} pirate')
    pirate_loss = M.loss_on(base, M.texts('pirate')[1])
    print(f'loss on pirate text: {pirate_loss:.2f}   (it has never seen arr, ye, be or matey)')
    return {'log': log, 'total': total, 'in_w': in_w, 'loss': loss, 'tally': tally, 'pirate_loss': pirate_loss}


if __name__ == '__main__':
    run()
