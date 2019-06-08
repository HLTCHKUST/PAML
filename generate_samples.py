from utils.data_reader import Personas
from model.seq2seq import SeqToSeq
from model.transformer import Transformer
# from model.common_layer import evaluate
from utils.beam_omt import Translator
import pickle
from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import numpy as np 
from random import shuffle
from copy import deepcopy
import math

def do_learning(model, train_iter, val_iter, iterations, logger):
    for i in range(1,iterations):
        for j, d in enumerate(train_iter):
            _, _, _ = model.train_one_batch(d)
    evaluate(model, val_iter, logger=logger)
    return logger


def evaluate(model, data, logger):
    t = Translator(model, model.vocab)
    for j, batch in enumerate(data):
        loss, ppl, _ = model.train_one_batch(batch, train=False)
        sent_g = model.decoder_greedy(batch)
        sent_b, _ = t.translate_batch(batch)
        for i, sent in enumerate(sent_g):
            new_words = []
            for w in sent_b[i][0]:
                if w==config.EOS_idx:
                    break
                new_words.append(w)
            sent_beam_search = ' '.join([model.vocab.index2word[idx] for idx in new_words])
            logger.append({"dialog":batch['input_txt'][i],"answer":sent_beam_search,"quality":[loss[i],math.exp(loss[i])]})
    return logger


p = Personas()
# Build model, optimizer, and set states
print("Test model",config.model)
model = Transformer(p.vocab,model_file_path=config.save_path,is_eval=False)

iterations = 11
weights_original = deepcopy(model.state_dict())
tasks = p.get_personas('test')
logger = []
for per in tqdm(tasks):
    num_of_dialog = p.get_num_of_dialog(persona=per, split='test')
    for val_dial_index in range(num_of_dialog):
        train_iter, val_iter = p.get_data_loader(persona=per,batch_size=config.batch_size, split='test', fold=val_dial_index)
        logger = do_learning(model, train_iter, val_iter, iterations=iterations, logger=logger)
        model.load_state_dict({ name: weights_original[name] for name in weights_original })

for l in logger:
    str_temp = "Dialogue History <\br>"
    for idx, d in enumerate(l["dialog"]):
        spk = 'Usr:' if idx % 2 == 0 else 'Sys:'
        str_temp += 'Usr: '+ d + "<\br>"
    str_temp += "<\br>"
    str_temp += "Answer<\br>"
    str_temp += 'Usr:' + l["answer"]
    print(str_temp)
    