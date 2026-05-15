# data/rank/

Populated externally by `trader forecast tokenize-rank` from the parent trader project. Vocab = 100 cross-sectional buckets: `0` = worst-performing stock that day, `99` = best. Each `{TICKER}.train.bin` / `{TICKER}.val.bin` is a chronological per-stock sequence of those bucket tokens (uint16).

The patched `train.py` auto-detects per-ticker `.bin` files via `glob`. Drives the `rank_percentage_tokenizer_gpt_v1` model in the parent project.
