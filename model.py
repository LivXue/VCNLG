from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from modules.sentence_tramsformer import *
from utils import misc as utils


class VCLM(nn.Module):
    def __init__(self, opt):
        super(VCLM, self).__init__()
        self.common_size = opt.common_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.output_txt_len = opt.output_txt_len
        self.img_fea_size = opt.img_fea_size
        self.output_vocab_size = opt.output_vocab_size
        self.use_know = opt.use_know

        self.bos_idx = opt.bos_idx
        self.eos_idx = opt.eos_idx
        self.pad_idx = opt.pad_idx

        self.sentence_transformer = BERTTransformer(opt, opt.bert_architecture)
        if self.use_know:
            self.knowledge_transformer = BERTTransformer(opt, "bert-base-uncased")
        #self.knowledge_transformer.fine_tuning = True
        self.sentence_generator = TFS_Head(opt)#GPT2_Head(opt)

        self.img_fea_module = nn.Sequential(nn.Linear(self.img_fea_size, self.common_size),
                                            nn.GELU(),
                                            nn.LayerNorm(self.common_size))
        img_proto = torch.Tensor(80, self.common_size)
        nn.init.normal_(img_proto)
        self.img_proto = nn.Parameter(img_proto.unsqueeze(0))
        self.img_distill = MultiHeadAttention(opt.common_size, opt.common_size, opt.num_head, dropout=opt.drop_prob_lm)
        txt_proto = torch.Tensor(40, self.common_size)
        nn.init.normal_(txt_proto)
        self.txt_proto = nn.Parameter(txt_proto.unsqueeze(0))
        self.txt_distill = MultiHeadAttention(opt.common_size, opt.common_size, opt.num_head, dropout=opt.drop_prob_lm)
        if self.use_know:
            klg_proto = torch.Tensor(40, self.common_size)
            nn.init.normal_(klg_proto)
            self.klg_proto = nn.Parameter(klg_proto.unsqueeze(0))
            self.klg_distill = MultiHeadAttention(opt.common_size, opt.common_size, opt.num_head, dropout=opt.drop_prob_lm)

    def get_ctx_feature(self, ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, know_ids, know_mask):
        b_s = ctx_img.size(0)
        txt_features = self.sentence_transformer(ctx_text, ctx_text_mask)
        txt_features, _ = self.txt_distill(self.txt_proto.expand(b_s, -1, -1), txt_features, txt_features,
                                           ctx_text_mask.unsqueeze(1).bool())
        img_features = self.img_fea_module(ctx_img)
        img_features, _ = self.img_distill(self.img_proto.expand(b_s, -1, -1), img_features, img_features,
                                           ctx_img_mask.unsqueeze(1).bool())
        if self.use_know:
            klg_features = self.knowledge_transformer(know_ids, know_mask)
            klg_features, _ = self.klg_distill(self.klg_proto.expand(b_s, -1, -1), klg_features, klg_features,
                                               know_mask.unsqueeze(1).bool())
        else:
            klg_features = None
        return img_features, txt_features, klg_features

    def _forward(self, ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask, input_ids, reuse_features=None, return_feature=False):
        # Project to common feature space
        if reuse_features is not None:
            img_features, txt_features, klg_features, past = reuse_features
        else:
            img_features, txt_features, klg_features = self.get_ctx_feature(ctx_img, ctx_img_mask, ctx_text,
                                                                            ctx_text_mask,
                                                                            ctx_know, ctx_know_mask)
            past = None

        outputs, past = self.sentence_generator(input_ids, img_features, txt_features, klg_features)
        #outputs = F.log_softmax(outputs, dim=-1)

        if return_feature:
            return outputs, [img_features, txt_features, klg_features, [[None, p[1]] for p in past]]
        else:
            return outputs

    def _sample(self, ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask, greedy=True, reuse_features=None, return_feature=False):
        b_s = ctx_img.size(0)
        if reuse_features is not None:
            img_features, txt_features, klg_features, past = reuse_features
        else:
            img_features, txt_features, klg_features = self.get_ctx_feature(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask,
                                                                            ctx_know, ctx_know_mask)
            past = None

        seq = img_features.new_full((b_s, self.output_txt_len), self.pad_idx, dtype=torch.long)
        seqLogits = img_features.new_zeros(b_s, self.output_txt_len, self.output_vocab_size)

        word_idx = img_features.new_full([b_s, 1], self.bos_idx, dtype=torch.long)
        for t in range(self.output_txt_len):
            logits, past = self.sentence_generator(word_idx, img_features, txt_features, klg_features, past=past)
            #logprobs = F.log_softmax(logits[:, -1], dim=-1)
            logits = logits[:, -1]

            if greedy:
                word_idx = logits.argmax(dim=-1)
            else:
                filtered_scores = self.top_p(F.softmax(logits.detach(), -1))
                # Sampling
                word_idx = torch.multinomial(filtered_scores, num_samples=1).squeeze(1)

            if t == 0:
                unfinished = word_idx != self.eos_idx
            else:
                word_idx[~unfinished] = self.pad_idx
                logits = logits * unfinished.unsqueeze(1)
                unfinished = unfinished * (word_idx != self.eos_idx)
            seq[:, t] = word_idx
            seqLogits[:, t] = logits
            if unfinished.sum() == 0:
                break

        if return_feature:
            return seq, seqLogits, [img_features, txt_features, klg_features, [[None, p[1]] for p in past]]
        else:
            return seq, seqLogits

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    # Obtain top_p probabilities for nucleur sampling
    def top_p(self, probs, p=0.9):
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0)
        return probs

    def top_k(self, probs, k=50):
        topk_probs = torch.topk(probs, k=k, dim=-1)[0][..., -1, None]
        indices_to_remove = probs < topk_probs
        probs = probs.masked_fill(indices_to_remove, 0)
        return probs
