from utils import config
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
import os
import time
import torch


class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)

def dup_batch(batch, idx, dup_times):
    new_batch = {}
    input_len = batch["input_lengths"][idx]
    for key in ["input_batch", "target_batch"]:
        new_batch[key] = batch[key][:input_len, idx:idx+1].repeat(1, dup_times)

    if "input_ext_vocab_batch" in batch:
        for key in ["input_ext_vocab_batch", "target_ext_vocab_batch"]:
            new_batch[key] = batch[key][:input_len, idx:idx+1].repeat(1, dup_times)
        new_batch["article_oovs"] = [batch["article_oovs"][idx] for _ in range(dup_times)]
        new_batch["max_art_oovs"] =  batch["max_art_oovs"]

    for key in ["input_txt", "target_txt"]:
        new_batch[key] = [batch[key][idx] for _ in range(dup_times)]
    for key in ["input_lengths", "target_lengths"]:
        new_batch[key] = batch[key][idx:idx+1].repeat(dup_times)

    return new_batch

class BeamSearch(object):
    def __init__(self, model, lang):
        
        self.model = model
        self.lang = lang
        self.vocab_size = lang.n_words

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):

        batch_size = batch["input_lengths"].size(0)
        decoded_sents = []
    
        for i in range(batch_size):
            new_batch = dup_batch(batch, i, config.beam_size)
            enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = get_input_from_batch(new_batch)
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search_sample(enc_batch, enc_padding_mask, enc_lens, 
                                            enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            if config.pointer_gen:
                art_oovs = batch["article_oovs"][i]
                len_oovs = len(art_oovs)
                decoded_words = []
                for idx in output_ids:
                    if idx < self.vocab_size:
                        decoded_words.append(self.lang.index2word[idx])    
                    elif idx - self.vocab_size < len_oovs:
                        decoded_words.append(art_oovs[idx - self.vocab_size])
                    else:
                        raise ValueError("invalid output id")
            else:
                decoded_words = [self.lang.index2word[idx] for idx in output_ids]

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index('EOS')
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            decoded_sents.append(decoded_words)
        return decoded_sents

    def beam_search_sample(self, enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0):
        #batch should have only one example by duplicate
        
        encoder_outputs, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze(0)
        dec_c = dec_c.squeeze(0)
        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[config.SOS_idx],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_step and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab_size else config.UNK_idx \
                             for t in latest_tokens]
            y_t_1 = torch.LongTensor(latest_tokens)
            if config.USE_CUDA:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps, training=False)

            topk_log_probs, topk_ids = torch.topk(final_dist, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == config.EOS_idx:
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

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
