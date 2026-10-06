"""The tiny language model: it reads the last 8 words and scores every word for "what comes next".

8 words -> 8 x 64 = 512 numbers -> W1 -> 1024 -> W2 -> 1024 -> W3 -> 64 scores, one per word.
Almost all of its numbers sit in three tables: the weight matrices W1, W2 and W3.
"""
import os

import numpy as np

from language import ID, STYLES, VOCAB, corpus, style_of

np.seterr(divide='ignore', over='ignore', invalid='ignore')   # numpy 2 on Apple chips prints false matmul warnings

N_CTX, EMB, HID = 8, 64, 1024
HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ['W1', 'W2', 'W3']


def new_model(seed=0):
    """Small random numbers everywhere. Each W has one row per output and one column per input."""
    rng = np.random.default_rng(seed)
    V, inp = len(VOCAB), N_CTX * EMB
    f = np.float32
    return {
        'E': (rng.normal(0, 0.3, (V, EMB))).astype(f),                    # one row of 64 numbers per word
        'W1': (rng.normal(0, 1, (HID, inp)) * np.sqrt(2 / inp)).astype(f), 'b1': np.zeros(HID, f),
        'W2': (rng.normal(0, 1, (HID, HID)) * np.sqrt(2 / HID)).astype(f), 'b2': np.zeros(HID, f),
        'W3': (rng.normal(0, 1, (V, HID)) * np.sqrt(1 / HID)).astype(f), 'b3': np.zeros(V, f),
    }


def count(params):
    """How many numbers are in a dictionary of arrays."""
    return int(sum(v.size for v in params.values()))


def forward(model, ctx, side=None):
    """Scores for the next word, for a batch of contexts. `side` is an optional adapter (step 5).

    Returns the scores and the input of every layer, which backward() needs.
    """
    x = model['E'][ctx].reshape(len(ctx), -1)
    xs = [x]
    for name in LAYERS:
        y = x @ model[name].T + model['b' + name[1]]
        if side is not None:
            y = y + side.forward(name, x)          # the adapter's path runs next to the matrix
        x = y if name == 'W3' else np.maximum(y, 0)
        xs.append(x)
    return x, xs


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def backward(model, ctx, xs, nxt, side=None, base=True):
    """The loss, and for every weight the gradient: how the loss changes if that weight grows.

    base=False skips the gradients of the model's own weights (they are frozen).
    """
    n = len(nxt)
    p = softmax(xs[-1])
    loss = float(-np.log(p[np.arange(n), nxt] + 1e-12).mean())
    d = p
    d[np.arange(n), nxt] -= 1
    d /= n
    g = {}
    for i in (2, 1, 0):
        name, x = LAYERS[i], xs[i]
        if i < 2:
            d = d * (xs[i + 1] > 0)
        if base:
            g[name] = d.T @ x
            g['b' + name[1]] = d.sum(axis=0)
        dx = d @ model[name]
        if side is not None:
            dx = dx + side.backward(name, x, d)
        d = dx
    if base:
        g['E'] = np.zeros_like(model['E'])
        np.add.at(g['E'], ctx, d.reshape(n, N_CTX, EMB))
    return loss, g


class Adam:
    """The usual optimizer. It keeps two extra numbers for every number it trains."""

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t, self.m, self.v = lr, b1, b2, eps, 0, {}, {}

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            m = self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g
            v = self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g * g
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            params[k] -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(params[k].dtype)

    def memory(self):
        """How many numbers the optimizer itself is holding."""
        return int(sum(a.size for a in self.m.values()) + sum(a.size for a in self.v.values()))


def windows(words):
    """Every stretch of 8 words, and the word that follows it."""
    idx = np.arange(N_CTX)[None, :] + np.arange(len(words) - N_CTX)[:, None]
    return words[idx], words[N_CTX:]


def train(model, words, steps, lr, side=None, batch=256, seed=0, log=None):
    """Train on a text. Without `side`, every weight of the model changes.

    With `side` (an adapter), the model is frozen and only the adapter's numbers change.
    """
    ctx, nxt = windows(words)
    rng = np.random.default_rng(seed)
    opt = Adam(lr)
    for step in range(steps):
        pick = rng.integers(0, len(nxt), batch)
        scores, xs = forward(model, ctx[pick], side)
        loss, g = backward(model, ctx[pick], xs, nxt[pick], side, base=side is None)
        if side is None:
            opt.step(model, g)
        else:
            opt.step(side.p, side.g)
        if log is not None and (step % 20 == 0 or step == steps - 1):
            log.append((step, loss))
    return opt


def loss_on(model, words, side=None, limit=20000):
    """Average loss on a text the model did not train on. Lower is better."""
    ctx, nxt = windows(words[:limit])
    scores, xs = forward(model, ctx, side)
    return backward(model, ctx, xs, nxt, base=False)[0]


def generate(model, start, n_sentences, seed=0, side=None, max_words=24):
    """Write sentences one word at a time: score, turn scores into chances, pick, repeat."""
    rng = np.random.default_rng(seed)
    ctx = list(start[-N_CTX:])
    out = []
    for _ in range(n_sentences):
        words = []
        while len(words) < max_words:
            p = softmax(forward(model, np.array([ctx]), side)[0])[0].astype(np.float64)
            w = int(rng.choice(len(VOCAB), p=p / p.sum()))
            words.append(VOCAB[w])
            ctx = ctx[1:] + [w]
            if VOCAB[w] == '.':
                break
        out.append(words)
    return out


def styles(sentences):
    """Count how many sentences are correct in each style."""
    tally = {'plain': 0, 'pirate': 0, 'polite': 0, 'wrong': 0}
    for s in sentences:
        tally[style_of(s) or 'wrong'] += 1
    return tally


def start(style='plain'):
    """Eight words to continue from: the end of one sentence in that style."""
    return [ID[w] for w in STYLES[style]('the old sailor finds the big net .'.split())][-N_CTX:]


def texts(style):
    """The text to train on and a text to test on, in one style. Plain is big; a style has only 3,000 sentences."""
    n, a, b = {'plain': (100000, 0, 1), 'pirate': (3000, 2, 3), 'polite': (3000, 4, 5)}[style]
    return corpus(n, style, a), corpus(1000, style, b)


def get_base(log=None):
    """The model that knows the plain language. Trained once (about a minute), then loaded from base.npz."""
    path = os.path.join(HERE, 'base.npz')
    if not os.path.exists(path):
        model, curve = new_model(0), []
        train(model, texts('plain')[0], 5000, 1e-3, log=curve)
        np.savez(path, _log=np.array(curve), **model)
    saved = dict(np.load(path))
    if log is not None:
        log.extend(map(tuple, saved['_log'].tolist()))
    return {k: v for k, v in saved.items() if k != '_log'}


def clone(model):
    return {k: v.copy() for k, v in model.items()}


def write(model, n=3, seed=1, side=None, style='plain'):
    return [' '.join(s) for s in generate(model, start(style), n, seed, side)]
