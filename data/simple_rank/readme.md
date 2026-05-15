# data/simple_rank/

Populated externally by `trader forecast tokenize-simple-rank --bucket-width N` from the parent trader project. Tokens encode each stock's cross-sectional ordinal rank by daily return: stocks sorted DESCENDING by `open[t+1]/open[t] - 1`, then `bucket = rank // bucket_width`. Token `0` = best-performing stock that day.

Vocab size is whatever falls out of the data — `meta.json` records the trained vocab; the CLI prints it for you to copy into `config/train_simple_rank.py`. Drives the `rank_simple_tokenizer_gpt_v1` model in the parent project.
