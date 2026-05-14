# Train tiny GPT on per-ticker joint return-bucket × volume-decile token streams.
# Data populated by `trader forecast tokenize-joint-rank` (see
# data/joint_rank/readme.md).

out_dir = 'out-joint-rank'
dataset = 'joint_rank'

batch_size = 1
block_size = 2048
gradient_accumulation_steps = 32

# Vocab = 100 return buckets × 10 volume deciles = 1000 joint tokens.
vocab_size = 1000

# Tiny GPT (~0.5M params, same shape as rank_gpt_v1).
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

device = 'mps'
compile = False
dtype = 'float32'
