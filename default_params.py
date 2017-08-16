hidden_dim = 256
emb_dim = 128
batch_size = 16
max_enc_steps = 400
max_dec_steps = 100
beam_size = 4
min_dec_steps = 35
vocab_size = 50000
training_steps = 240000
lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0
cov_loss_wt = 1.0
pointer_gen = True
coverage = False
convert_to_coverage_model = False
mode = 'train'
