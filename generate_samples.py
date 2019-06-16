import matplotlib
matplotlib.use('Agg')
from utils.data_reader import Personas
from model.transformer import Transformer
import pickle
from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
from utils.beam_omt import Translator
import os
import time
import numpy as np 
from random import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import math

def generate(model, data, persona):
    t = Translator(model, model.vocab)
    for j, batch in enumerate(data):
        _, _, _ = model.train_one_batch(batch, train=False)
        sent_b, _ = t.translate_batch(batch)
        for i in range(len(batch["target_txt"])):
            new_words = []
            for w in sent_b[i][0]:
                if w==config.EOS_idx:
                    break
                new_words.append(w)
                if len(new_words)>2 and (new_words[-2]==w):
                    new_words.pop()
            sent_beam_search = ' '.join([model.vocab.index2word[idx] for idx in new_words])
            print("----------------------------------------------------------------------")
            print("----------------------------------------------------------------------")
            print("persona set")
            print(pp.pformat(persona))
            print("dialogue context:")
            print(pp.pformat(batch['input_txt'][i]))
            print("Beam: {}".format(sent_beam_search))
            print("Ref:{}".format(batch["target_txt"][i]))
            print("----------------------------------------------------------------------")
            print("----------------------------------------------------------------------")

def do_learning(model, train_iter, val_iter, iterations, persona):
    for i in range(1,iterations):
        for j, d in enumerate(train_iter):
            _, _, _ = model.train_one_batch(d)
    generate(model, val_iter, persona)


p = Personas()
# Build model, optimizer, and set states
print("Test model",config.model)
model = Transformer(p.vocab,model_file_path=config.save_path,is_eval=False)
# get persona map
filename = 'data/ConvAI2/test_persona_map'
with open(filename,'rb') as f:
    persona_map = pickle.load(f)

#generate
iterations = 11
weights_original = deepcopy(model.state_dict())
tasks = p.get_personas('test')
for per in tqdm(tasks):
    num_of_dialog = p.get_num_of_dialog(persona=per, split='test')
    for val_dial_index in range(num_of_dialog):
        train_iter, val_iter = p.get_data_loader(persona=per,batch_size=config.batch_size, split='test', fold=val_dial_index)
        persona=[]
        for ppp in persona_map[per]:
            persona+=ppp
        persona = list(set(persona))
        do_learning(model, train_iter, val_iter, iterations=iterations, persona=persona)
        model.load_state_dict({ name: weights_original[name] for name in weights_original })
