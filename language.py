"""The made-up language the model learns, and two styles of it that we teach later.

Plain:   the old sailor finds the net .
Pirate:  arr , ye old sailor finds ye net , matey .
Polite:  please , the old sailor kindly finds the net , thank you .
"""
import re

import numpy as np

WHO = ['sailor', 'captain', 'farmer', 'baker', 'teacher', 'doctor', 'king', 'queen', 'robot', 'pilot', 'singer', 'wolf']
WHAT = ['ship', 'net', 'bread', 'book', 'sword', 'crown', 'rocket', 'song', 'map', 'drum', 'cake', 'lamp']
WHERE = ['harbor', 'farm', 'kitchen', 'school', 'castle', 'moon', 'forest', 'market']
DOES = ['finds', 'sees', 'likes', 'wants', 'carries', 'paints', 'guards', 'sells']
ADJ = ['big', 'small', 'old', 'new', 'quiet', 'loud', 'happy', 'tired']
FUNC = ['the', 'a', 'and', 'in', 'near', 'is', ',', '.']
PIRATE = ['arr', 'ye', 'be', 'matey']            # the plain text never uses these four words
POLITE = ['please', 'kindly', 'thank', 'you']    # nor these four
VOCAB = FUNC + ADJ + DOES + WHO + WHAT + WHERE + PIRATE + POLITE
ID = {w: i for i, w in enumerate(VOCAB)}
assert len(ID) == len(VOCAB) == 64
POOLS = {'who': WHO, 'what': WHAT, 'where': WHERE, 'does': DOES, 'adj': ADJ}

# every plain sentence follows one of five patterns; "adj?" is an adjective four times out of ten
PATTERNS = [
    'the adj? who does the adj? what .',
    'the who does a adj? what near the where .',
    'in the where , the who does a what .',
    'the what is adj and the where is adj .',
    'a who and a who does the what .',
]


def plain(rng):
    """One plain sentence, as a list of words."""
    out = []
    for slot in PATTERNS[rng.integers(len(PATTERNS))].split():
        if slot == 'adj?':
            if rng.random() > 0.4:
                continue
            slot = 'adj'
        out.append(POOLS[slot][rng.integers(len(POOLS[slot]))] if slot in POOLS else slot)
    return out


def pirate(words):
    """arr at the front, ye for the, be for is, matey at the end."""
    swap = {'the': 'ye', 'is': 'be'}
    return ['arr', ','] + [swap.get(w, w) for w in words[:-1]] + [',', 'matey', '.']


def polite(words):
    """please at the front, kindly before the verb, thank you at the end."""
    body = []
    for w in words[:-1]:
        body += ['kindly', w] if w in DOES else [w]
    return ['please', ','] + body + [',', 'thank', 'you', '.']


STYLES = {'plain': lambda s: s, 'pirate': pirate, 'polite': polite}


def corpus(n_sentences, style='plain', seed=0):
    """A long list of word numbers: n sentences in one style, one after another."""
    rng = np.random.default_rng(seed)
    words = [w for _ in range(n_sentences) for w in STYLES[style](plain(rng))]
    return np.array([ID[w] for w in words], dtype=np.int64)


def _regex(pattern):
    parts = []
    for slot in pattern.split():
        if slot == 'adj?':
            parts.append('(?:(?:%s) )?' % '|'.join(ADJ))
        elif slot in POOLS:
            parts.append('(?:%s) ' % '|'.join(POOLS[slot]))
        else:
            parts.append(re.escape(slot) + ' ')
    return re.compile(''.join(parts).rstrip(' ') + '$')


_PLAIN = [_regex(p) for p in PATTERNS]


def is_plain(words):
    """True if the words are one correct plain sentence."""
    text = ' '.join(words)
    return any(r.match(text) for r in _PLAIN)


def style_of(words):
    """'plain', 'pirate' or 'polite' if the sentence is correct in that style, else None."""
    if is_plain(words):
        return 'plain'
    if words[:2] == ['arr', ','] and words[-3:] == [',', 'matey', '.'] and 'the' not in words and 'is' not in words:
        back = {'ye': 'the', 'be': 'is'}
        if is_plain([back.get(w, w) for w in words[2:-3]] + ['.']):
            return 'pirate'
    if words[:2] == ['please', ','] and words[-4:] == [',', 'thank', 'you', '.']:
        body, inner = [], words[2:-4]
        for i, w in enumerate(inner):
            if w in DOES and (i == 0 or inner[i - 1] != 'kindly'):
                return None
            if w != 'kindly':
                body.append(w)
            elif i + 1 == len(inner) or inner[i + 1] not in DOES:
                return None
        if is_plain(body + ['.']):
            return 'polite'
    return None


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    for _ in range(3):
        s = plain(rng)
        print(' '.join(s)); print('   ', ' '.join(pirate(s))); print('   ', ' '.join(polite(s)))
    print(len(VOCAB), 'words')
