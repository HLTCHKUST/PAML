### MOSTO OF IT TAKEN FROM https://github.com/kolloldas/torchnlp
## MINOR CHANGES
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
from utils import config
from utils.metric import rouge, moses_multi_bleu, _prec_recall_f1_score, entailtment_score
from utils.beam_omt import Translator
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
from utils.load_bert import bert_model 

class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding = 'both', 
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        # self.layer_norm_end = LayerNorm(hidden_size)
        
    def forward(self, inputs, mask=None):
        x = inputs
        
        # Layer Normalization
        x_norm = self.layer_norm_mha(x)
        
        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)
        
        # Dropout and residual
        x = self.dropout(x + y)
        
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        
        # Dropout and residual
        y = self.dropout(x + y)
        
        # y = self.layer_norm_end(y)
        return y

class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(DecoderLayer, self).__init__()
        
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, bias_mask, attention_dropout)

        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, None, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding = 'left', 
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        # self.layer_norm_end = LayerNorm(hidden_size)

        
    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask
        
        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)
        
        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)
        
        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)
        
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        
        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)
        
        # y = self.layer_norm_end(y)
        
        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight, mask



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, 
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from 
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            
        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5 ## sqrt
        self.bias_mask = bias_mask
        
        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
    
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)
        
    def forward(self, queries, keys, values, mask):
        
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        
        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        
        # Scale queries
        queries *= self.query_scale
        
        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)


        # # Add bias to mask future values
        # if self.bias_mask is not None:
        #     logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)
        
        ## attention weights 
        attetion_weights = logits.sum(dim=1)/self.num_heads

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)


        # Dropout
        weights = self.dropout(weights)
        
        # Combine with values to get context
        contexts = torch.matmul(weights, values)
        
        # Merge heads
        contexts = self._merge_heads(contexts)
        #contexts = torch.tanh(contexts)
        
        # Linear to get output
        outputs = self.output_linear(contexts)
        
        return outputs, attetion_weights

class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data), 
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size//2, (kernel_size - 1)//2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """
    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data), 
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()
        
        layers = []
        sizes = ([(input_depth, filter_size)] + 
                 [(filter_size, filter_size)]*(len(layer_config)-2) + 
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
    
    return torch_mask.unsqueeze(0).unsqueeze(1)

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if(config.USE_CUDA):
        return subsequent_mask.cuda()
    else:
        return subsequent_mask


class OutputLayer(nn.Module):
    """
    Abstract base class for output layer. 
    Handles projection to output labels
    """
    def __init__(self, hidden_size, output_size):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))


class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer
    """
    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        _, predictions = torch.max(probs, dim=-1)

        return predictions

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))

def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    # encoding[:, -1] = 1.0
    return np.transpose(encoding)

def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01 
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file).readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings


def share_embedding(vocab, pretrain=True):
    embedding = nn.Embedding(vocab.n_words, config.emb_dim)
    if(pretrain):
        pre_embedding = gen_embeddings(vocab)
        embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.weight.data.requires_grad = True
    return embedding

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(config.PAD_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_input_from_batch(batch):
    enc_batch = batch["input_batch"].transpose(0,1)
    enc_lens = batch["input_lengths"]
    batch_size, max_enc_len = enc_batch.size()
    assert enc_lens.size(0) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"].transpose(0,1)
        # max_art_oovs is the max over all the article oov list in the batch
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size())

    if config.USE_CUDA:
        if enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        c_t_1 = c_t_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch):

    dec_batch = batch["target_batch"].transpose(0,1)

    if(config.pointer_gen):
        target_batch = batch["target_ext_vocab_batch"].transpose(0,1)
    else:
        target_batch = dec_batch
        
    dec_lens_var = batch["target_lengths"]
    max_dec_len = max(dec_lens_var)

    assert max_dec_len == target_batch.size(1)

    dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def print_all(dial,ref,hyp_b,max_print):
    for i in range(len(ref)):
        print(pp.pformat(dial[i]))
        print("Beam: {}".format(hyp_b[i]))
        print("Ref:{}".format(ref[i]))
        print("----------------------------------------------------------------------")
        print("----------------------------------------------------------------------")
        if(i>max_print):break

bert = bert_model()
def evaluate(model, data, model_name='trs', ty='valid', writer=None, n_iter=0, ty_eval="before", verbose=False ):

    dial,ref, hyp_b= [],[],[]
    t = Translator(model, model.vocab)

    l = []
    p = []
    ent_b = []
    
    pbar = tqdm(enumerate(data),total=len(data))
    for j, batch in pbar:
        loss, ppl, _ = model.train_one_batch(batch, train=False)
        l.append(loss)
        p.append(ppl)
        if((j<3 and ty != "test") or ty =="test"): 

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
                hyp_b.append(sent_beam_search)
                ref.append(batch["target_txt"][i])
                dial.append(batch['input_txt'][i])
                ent_b.append(bert.predict_label([sent_beam_search for _ in range(len(batch['persona_txt'][i]))], batch['persona_txt'][i]))

        pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),np.mean(p)))
        if(j>4 and ty=="train"): break
    loss = np.mean(l)
    ppl = np.mean(p)
    ent_b = np.mean(ent_b)
    bleu_score_b = moses_multi_bleu(np.array(hyp_b), np.array(ref), lowercase=True)
 
    if(verbose):
        print("----------------------------------------------------------------------")
        print("----------------------------------------------------------------------")
        print_all(dial,ref,hyp_b,max_print=3 if ty != "test" else 100000000 )
        print("EVAL\tLoss\tPeplexity\tEntl_b\tBleu_b")
        print("{}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}".format(ty,loss,ppl,ent_b,bleu_score_b))
    return loss,ppl,ent_b,bleu_score_b
