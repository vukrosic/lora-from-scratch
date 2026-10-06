# LoRA from scratch

LoRA in plain numpy, built in eight steps: a tiny language model, a full fine-tune, the low-rank idea, the adapter layer with its own gradients, training, merging and swapping.
12 Python files, 840 lines, no deep-learning library.

Companion code for the video "How LoRA Fine-Tunes With 1% of the Weights".

## Run it

```bash
pip install numpy
bash run_all.sh
```

That runs every step and the exercise (`demo-output.txt` is one saved run). The first run trains the base model
(about a minute) and saves it as `base.npz`; the whole script takes about four minutes the first time. Python 3.9+.

```bash
python3 tests.py
```

## What it shows

The model reads the last 8 words of a made-up 64-word language and scores every word for "what comes next".
It has 1,644,608 numbers; 99.6% of them sit in three matrices, W1, W2 and W3. It learns the plain language first.
Then we teach it a style ("arr , ye old sailor finds ye net , matey ."), twice:

|  | full fine-tune | LoRA rank 4 |
| --- | --- | --- |
| numbers trained | 1,644,608 | 18,688 (1.14%) |
| loss on pirate text | 0.893 | 0.875 |
| correct sentences of 1,000 | 994 | 977 |
| optimizer memory (numbers) | 3,289,216 | 37,376 |
| file for the style (bytes) | 6,580,104 | 76,436 |
| plain text afterwards (loss) | 1.96 | 1.253, the base is untouched |

100 styles as full copies take 658.4 MB. One base and 100 adapters take 14.2 MB.

## The steps

| File | What it adds |
| --- | --- |
| `language.py` | The made-up language, the pirate and polite styles, and a checker that says if a sentence is correct |
| `model.py` | The model, its gradients, the Adam optimizer, training and writing sentences |
| `step1_base.py` | Train the base model on plain text and count its numbers |
| `step2_full_finetune.py` | Fine-tune every number on pirate text; what that costs and what the copy forgets |
| `step3_the_change.py` | Subtract the base from the fine-tuned copy and look at the change in each matrix |
| `step4_low_rank.py` | A column times a row; a change of rank r costs r × (rows + columns) numbers |
| `step5_lora_layer.py` | The adapter: W stays frozen, a side path B·A is added, B starts at zero |
| `step6_train_lora.py` | Train only A and B, and compare with the full fine-tune |
| `step7_merge_swap.py` | Fold B·A into W; swap pirate and polite adapters on one base; file sizes |
| `step8_rank.py` | The two settings: rank 1 to 16, alpha 1 to 128 |
| `exercise.py` | The exercise: an adapter on only one of the three matrices |
| `tests.py` | Checks for every step, including the adapter's gradients against nudging one number |

## The exercise

The adapter sits on all three matrices and writes 977 correct pirate sentences of 1,000.
Put it on only one matrix. Which one matters most? Guess first, then run:

```bash
python3 exercise.py
```

## What is different from real LoRA use

- The model is a small network of three matrices, not a transformer. In the paper the adapters sit on attention
  matrices (the query and value projections).
- The language has 64 words and five sentence patterns, so "correct" can be checked exactly.
- Newer variants are not built here: QLoRA (2023), DoRA (2024), AdaLoRA (2023).

## Source

Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen: "LoRA: Low-Rank Adaptation of Large Language Models", 2021,
[arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

MIT License.
