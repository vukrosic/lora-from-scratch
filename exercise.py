"""Exercise. Our adapter sits on all three matrices.  Put a rank-4 adapter on one matrix only.

Which single matrix teaches the pirate style best: W1, W2 or W3?  Guess first, then run:  python3 exercise.py
Hint: train_adapter() takes layers=['W2'].
"""
import model as M
from step6_train_lora import train_adapter


def run():
    base = M.get_base()
    pirate, test = M.texts('pirate')
    rows = []
    for layers in (['W1'], ['W2'], ['W3'], ['W1', 'W2', 'W3']):
        side = train_adapter(base, pirate, layers=layers)[0]
        ok = M.styles(M.generate(base, M.start('pirate'), 1000, seed=5, side=side))['pirate']
        rows.append({'on': ' '.join(layers), 'n': M.count(side.p), 'loss': M.loss_on(base, test, side), 'ok': ok})
        print(f'adapter on {rows[-1]["on"]:<9} {rows[-1]["n"]:>6,} numbers   loss {rows[-1]["loss"]:.3f}   correct {ok} of 1,000')
    return rows


if __name__ == '__main__':
    run()
