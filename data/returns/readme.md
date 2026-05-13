# returns

Daily-return token sequences for nanoGPT, written by the parent trader project
(not committed — regenerable). Vocab is 1002 tokens: 1000 bins of 0.1% width
covering [-50.05%, +49.95%), plus two overflow tokens.

Populate from the trader repo root:

    .venv/bin/trader forecast tokenize

This writes one `{TICKER}.train.bin` and `{TICKER}.val.bin` per cached S&P 500
ticker (uint16, raw little-endian), plus `meta.json` with the train/val cutoff
and aggregate counts.

Train with:

    python train.py config/train_returns.py
