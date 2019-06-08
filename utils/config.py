import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--persona", action="store_true")
parser.add_argument("--hidden_dim", type=int, default=100)
parser.add_argument("--emb_dim", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=2.0)
parser.add_argument("--max_enc_steps", type=int, default=400)
parser.add_argument("--max_dec_steps", type=int, default=20)
parser.add_argument("--min_dec_steps", type=int, default=5)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--save_path", type=str, default="save/")
parser.add_argument("--save_path_dataset", type=str, default="save/")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--pointer_gen", action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="trs")
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true")
parser.add_argument("--noam", action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)


## transformer 
parser.add_argument("--hop", type=int, default=6)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)

#meta
parser.add_argument("--fix_dialnum_train", action="store_true")
parser.add_argument("--meta_lr", type=float, default=0.1)
parser.add_argument("--mate_interation", type=int, default=1)
parser.add_argument("--use_sgd", action="store_true")
parser.add_argument('--meta_batch_size', type=int, default=1)
parser.add_argument("--meta_optimizer", type=str, default="sgd")
parser.add_argument("--load_frompretrain", type=str, default="None")
parser.add_argument("--k_shot", type=int, default=20)

arg = parser.parse_args()
print(arg)
model = arg.model
persona = arg.persona


# Hyperparameters
hidden_dim= arg.hidden_dim
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr

max_enc_steps=arg.max_enc_steps
max_dec_step= max_dec_steps=arg.max_dec_steps

min_dec_steps=arg.min_dec_steps 
beam_size=arg.beam_size

adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=arg.max_grad_norm

USE_CUDA = arg.cuda
pointer_gen = arg.pointer_gen
is_coverage = arg.is_coverage
use_oov_emb = arg.use_oov_emb
cov_loss_wt = 1.0
lr_coverage=0.15
eps = 1e-12
epochs = 10000
UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3


emb_file = "vectors/glove.6B.{}d.txt".format(str(emb_dim))
preptrained = arg.pretrain_emb

save_path = arg.save_path
save_path_dataset = arg.save_path_dataset

test = arg.test
if(not test):
    save_path_dataset = save_path


### transformer 
hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter


label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight


## Meta-learn
meta_lr = arg.meta_lr
meta_iteration = arg.mate_interation
use_sgd = arg.use_sgd
meta_batch_size = arg.meta_batch_size
meta_optimizer = arg.meta_optimizer
fix_dialnum_train = arg.fix_dialnum_train
load_frompretrain = arg.load_frompretrain
k_shot = arg.k_shot