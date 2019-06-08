from utils.data_reader import Personas
from model.seq2seq import SeqToSeq
from model.transformer import Transformer
from utils.beam_omt import Translator
from utils import config
from utils.data_reader import Dataset,collate_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time 
import ast



def make_batch(inp,vacab):
    temp = [[inp,['',''],0]]
    d = Dataset(temp,vacab)
    loader = torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return iter(loader).next()

p = Personas()
persona = ast.literal_eval(p.get_task('train'))
print(persona)
model = Transformer(p.vocab,model_file_path=config.save_path,is_eval=True)
t = Translator(model, p.vocab)
print('Start to chat')
while(True):
    msg = input(">>> ")
    if(len(str(msg).rstrip().lstrip()) != 0):
        persona +=  [str(msg).rstrip().lstrip()]
        batch = make_batch(persona, p.vocab)
        sent_b, batch_scores = t.translate_batch(batch)
        ris = ' '.join([p.vocab.index2word[idx] for idx in sent_b[0][0]]).replace('EOS','').rstrip().lstrip()
        print(">>>",ris)
        persona += [ris]