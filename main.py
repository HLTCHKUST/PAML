from utils.data_reader import Personas
from model.transformer import Transformer
from model.common_layer import evaluate
from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time 
import numpy as np 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

p = Personas()

data_loader_tr, data_loader_val, data_loader_test = p.get_all_data(batch_size=config.batch_size)

if(config.test):
    print("Test model",config.model)
    model = Transformer(p.vocab,model_file_path=config.save_path,is_eval=True)
    evaluate(model,data_loader_test,model_name=config.model,ty='test')
    exit(0)

model = Transformer(p.vocab)
print("MODEL USED",config.model)
print("TRAINABLE PARAMETERS",count_parameters(model))

best_ppl = 1000
cnt = 0
for e in range(config.epochs):
    print("Epoch", e)
    p, l = [],[]
    pbar = tqdm(enumerate(data_loader_tr),total=len(data_loader_tr))
    for i, d in pbar:
        loss, ppl, _ = model.train_one_batch(d)
        l.append(loss)
        p.append(ppl)
        pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),np.mean(p)))
    loss,ppl_val,ent_b,bleu_score_b = evaluate(model,data_loader_val,model_name=config.model,ty="valid")
    if(ppl_val <= best_ppl):
        best_ppl = ppl_val
        cnt = 0
        model.save_model(best_ppl,e,0,0,0,ent_b)
    else: 
        cnt += 1
    if(cnt > 10): break



