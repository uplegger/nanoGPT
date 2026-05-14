# data/joint_rank/

Populated externally by `trader forecast tokenize-joint-rank` from the parent trader project.

**Vocab = 1000** joint tokens: `token = return_bucket (0..99) * 10 + volume_decile (0..9)`. The return bucket follows the same cross-sectional spread scheme as `data/rank/`; the volume decile is the stock's cross-sectional volume rank that day (bottom 10% → 0, top 10% → 9).

Each `{TICKER}.train.bin` / `{TICKER}.val.bin` is a chronological per-stock sequence of these joint tokens (uint16). Drives the `rank_vol_gpt_v1` model in the parent project.
