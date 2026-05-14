# Train tiny GPT on per-ticker cross-sectional rank-bucket token streams.
# Data populated by `trader forecast tokenize-rank` (see data/rank/readme.md).

out_dir = 'out-rank'
dataset = 'rank'

# Per-ticker variable-length batches: get_batch yields one ticker at a time
# (B=1, T=ticker_length). gradient_accumulation_steps is the only knob for
# effective batch size.
batch_size = 1
block_size = 2048
gradient_accumulation_steps = 32

# Vocab = 100 cross-sectional buckets (0 = worst that day, 99 = best).
vocab_size = 100

# Tiny GPT (~0.5M params).
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 1000
warmup_iters = 50
lr_decay_iters = 1000
min_lr = 3e-5
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

eval_interval = 100
eval_iters = 50
log_interval = 20
always_save_checkpoint = False

# macOS Apple Silicon defaults.
device = 'mps'
compile = False
dtype = 'float32'
