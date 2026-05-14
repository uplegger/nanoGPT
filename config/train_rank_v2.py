# Train mid-sized GPT on per-ticker cross-sectional rank-bucket token streams.
# Reads the same data/rank/ files as train_rank.py (populated by
# `trader forecast tokenize-rank`); only the architecture and iter count
# differ.

out_dir = 'out-rank-v2'
dataset = 'rank'

batch_size = 1
block_size = 2048
gradient_accumulation_steps = 32

vocab_size = 100

# Mid-sized GPT (~3M params) — 6x the rank_gpt_v1 parameter count.
n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 3000
warmup_iters = 100
lr_decay_iters = 3000
min_lr = 3e-5
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

eval_interval = 200
eval_iters = 50
log_interval = 50
always_save_checkpoint = False

device = 'mps'
compile = False
dtype = 'float32'
