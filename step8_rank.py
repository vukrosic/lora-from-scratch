"""Step 8. The two dials of LoRA: the rank r (how many directions the change may use) and alpha (how loud it is)."""
import model as M
from step6_train_lora import train_adapter

RANKS = [1, 2, 4, 8, 16]
ALPHAS = [1, 4, 8, 32, 128]


def measure(base, test, r, alpha=None):
    side = train_adapter(base, M.texts('pirate')[0], r, alpha)[0]
    ok = M.styles(M.generate(base, M.start('pirate'), 1000, seed=5, side=side))['pirate']
    return {'r': r, 'alpha': side.alpha, 'n': M.count(side.p), 'loss': M.loss_on(base, test, side), 'ok': ok}


def run():
    base, test = M.get_base(), M.texts('pirate')[1]
    total = M.count(base)
    print('rank   numbers trained   share    loss on pirate text   correct of 1,000')
    ranks = []
    for r in RANKS:
        ranks.append(measure(base, test, r))
        m = ranks[-1]
        print(f'  {r:<3}   {m["n"]:>10,}     {m["n"] / total:>6.2%}        {m["loss"]:.3f}              {m["ok"]}')
    print('rank 4, different alpha:')
    alphas = []
    for alpha in ALPHAS:
        alphas.append(measure(base, test, 4, alpha))
        m = alphas[-1]
        print(f'  alpha {alpha:<4} (side path x {alpha / 4:g})   loss {m["loss"]:.3f}   correct {m["ok"]}')
    return {'ranks': ranks, 'alphas': alphas, 'total': total}


if __name__ == '__main__':
    run()
