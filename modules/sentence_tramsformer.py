import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from modules.gpt2.modeling_gpt2 import GPT2LMHeadModel

act = nn.GELU


class MultiHeadAttention(nn.Module):
    """
    multi-head attention layer with MEMORY module
    """

    def __init__(self, d_model, d_hidden, h, dropout=.1):
        """
        :param d_model: Output dimensionality of the model
        :param d_hidden: Dimensionality of queries, keys, and values
        :param h: Number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_hidden)
        self.fc_k = nn.Linear(d_model, h * d_hidden)
        self.fc_v = nn.Linear(d_model, h * d_hidden)
        self.fc_o = nn.Linear(h * d_hidden, d_model)
        self.res_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.h = h

        self.init_weights()

    def init_weights(self):
        #nn.init.xavier_uniform_(self.fc_q.weight)
        #nn.init.xavier_uniform_(self.fc_k.weight)
        #nn.init.xavier_uniform_(self.fc_v.weight)
        #nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.normal_(self.fc_q.weight, std=0.02)
        nn.init.normal_(self.fc_k.weight, std=0.02)
        nn.init.normal_(self.fc_v.weight, std=0.02)
        nn.init.normal_(self.fc_o.weight, std=0.02)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys=None, values=None, attention_mask=None, past=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :return:
        """
        assert (keys is not None and values is not None) or past is not None, "No input keys or values!"

        b_s, nq = queries.shape[:2]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_hidden).permute(0, 2, 1, 3).contiguous()  # (b_s, h, nq, d_hidden)
        if keys is not None:
            nk = keys.shape[1]
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_hidden).permute(0, 2, 1, 3).contiguous()  # (b_s, h, nk, d_hidden)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_hidden).permute(0, 2, 1, 3).contiguous()  # (b_s, h, nk, d_hidden)

        if past is not None and keys is not None:
            k = torch.cat((past[0], k), dim=-2).contiguous()
            v = torch.cat((past[1], v), dim=-2).contiguous()
        elif past is not None:
            k = past[0]
            v = past[1]

        if attention_mask is not None:
            attn_weight = softmax_one((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attention_mask.unsqueeze(1), dim=-1)
            attn_weight = torch.dropout(attn_weight, self.dropout, train=self.training)
            out = attn_weight @ v
            #out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1), dropout_p=self.dropout)
        else:
            attn_weight = softmax_one((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            attn_weight = torch.dropout(attn_weight, self.dropout, train=self.training)
            out = attn_weight @ v
            #out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_hidden)  # (b_s, nq, h*d_hidden)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        out = self.res_dropout(out)

        present = torch.stack((k, v))
        return out, present


def softmax_one(x, dim=None, _stacklevel=3):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_hidden, h, dropout=.1):
        super(TransformerLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_hidden, h, dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_model),
                                          nn.GELU(),
                                          nn.Linear(d_model, d_model),
                                          nn.Dropout(dropout))
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, input, mask=None, past=None):
        normed_input = self.ln_1(input)
        feature, present = self.self_att(normed_input, normed_input, normed_input, mask, past=past)
        feature = input + feature
        ml_feature = self.feed_forward(self.ln_2(feature))
        feature = ml_feature + feature

        return feature, present


class BERTTransformer(nn.Module):
    def __init__(self, opt, architecture="bert-large-uncased"):
        super(BERTTransformer, self).__init__()
        self.opt = opt
        self.act = act()
        self.bert_model = BertModel.from_pretrained(architecture)
        self.fine_tuning = False
        self.n_layers = self.bert_model.config.num_hidden_layers

        self.output_module = nn.Sequential(nn.Linear(self.bert_model.config.hidden_size, opt.common_size),
                                           self.act,
                                           nn.LayerNorm(opt.common_size))

        unfreezed_layer = ['layer.{}'.format(self.n_layers-1), 'layer.{}'.format(self.n_layers-2),
                           'layer.{}'.format(self.n_layers-3), 'layer.{}'.format(self.n_layers-4), 'pooler.', 'out.']
        for name, param in self.bert_model.named_parameters():
            param.requires_grad = False
            for ele in unfreezed_layer:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, input_ids, attention_mask):
        txt_features = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]

        if self.fine_tuning is False:
            txt_features = txt_features.detach()

        txt_features = self.output_module(txt_features)
        return txt_features


class TFS_Head(nn.Module):
    """
    Trained-from-scratch head
    """
    def __init__(self, opt):
        super(TFS_Head, self).__init__()
        self.opt = opt
        self.num_head = getattr(opt, 'num_head', 8)
        self.n_layers = 2

        self.act = act()
        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.word_emb = gpt2.transformer.wte
        #self.word_emb.weight.requires_grad = False
        self.dropout = nn.Dropout(opt.drop_prob_lm)
        self.trans_layers = nn.ModuleList([TransformerLayer(opt.common_size, opt.common_size, self.num_head,
                                                            dropout=opt.drop_prob_lm) for _ in range(self.n_layers)])
        self.cross_layers = nn.ModuleList([MultiHeadAttention(opt.common_size, opt.common_size, self.num_head,
                                                              dropout=opt.drop_prob_lm) for _ in range(self.n_layers)])
        self.output_module = nn.Sequential(nn.Linear(opt.common_size, 2 * opt.common_size),
                                           self.act,
                                           nn.LayerNorm(2 * opt.common_size),
                                           nn.Linear(2 * opt.common_size, opt.output_vocab_size))

        self.gen_position_emb = gpt2.transformer.wpe
        self.txt_position_emb = nn.Parameter(self.init_tensor(40, opt.common_size).unsqueeze(0))
        self.img_position_emb = nn.Parameter(self.init_tensor(80, opt.common_size).unsqueeze(0))
        self.klg_position_emb = nn.Parameter(self.init_tensor(40, opt.common_size).unsqueeze(0))

    def init_tensor(self, dim1, dim2):
        tmp = torch.Tensor(dim1, dim2)
        #nn.init.xavier_uniform_(tmp)
        nn.init.normal_(tmp, std=0.02)
        return tmp

    def forward(self, input_ids, img_feature, txt_features, klg_features=None, ctx_img_mask=None, ctx_text_mask=None, ctx_know_mask=None, past=None):
        # Get input embeddings
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        input_word_embs = self.word_emb(input_ids)

        # Prepare context embeddings
        seq_len = input_word_embs.shape[1]
        output_mask = torch.triu(torch.ones((seq_len, seq_len), device=input_word_embs.device, dtype=input_word_embs.dtype), diagonal=1).unsqueeze(0)

        # Get context masks
        if ctx_img_mask is None and ctx_text_mask is None and ctx_know_mask is None:
            ctx_mask = None
        else:
            if ctx_img_mask is None:
                ctx_img_mask = torch.ones(input_ids.size(0), img_feature.size(1), device=input_ids.device)
            if ctx_text_mask is None:
                ctx_text_mask = torch.ones(input_ids.size(0), txt_features.size(1), device=input_ids.device)
            if ctx_know_mask is None and klg_features is not None:
                ctx_know_mask = torch.ones(input_ids.size(0), klg_features.size(1), device=input_ids.device)

            ctx_mask = 1 - torch.cat((ctx_img_mask, ctx_text_mask, ctx_know_mask), dim=1)
            ctx_mask = ctx_mask.unsqueeze(1).float() * (-1e6)

        if past is not None and past[0][0] is not None:
            past_seq_len = past[0][0].shape[3]
            output_mask = torch.cat((torch.zeros((1, seq_len, past_seq_len), device=output_mask.device, dtype=input_word_embs.dtype), output_mask), dim=-1)
        else:
            past_seq_len = 0

        if past is not None and past[0][1] is not None:
            ctx_features = None
        else:
            img_feature = img_feature + self.img_position_emb[:, :img_feature.shape[1]]
            txt_features = txt_features + self.txt_position_emb[:, :txt_features.shape[1]]
            if klg_features is not None:
                klg_features = klg_features + self.klg_position_emb[:, :klg_features.shape[1]]
                ctx_features = torch.cat((img_feature, txt_features, klg_features), dim=1)
            else:
                ctx_features = torch.cat((img_feature, txt_features), dim=1)

        if past is None:
            past = [[None, None]] * self.n_layers

        output_mask = output_mask * (-1e6)
        # Input into Transformer layers
        input_pos_ids = torch.arange(past_seq_len, past_seq_len+seq_len, dtype=torch.long, device=input_ids.device)
        input_features = input_word_embs + self.gen_position_emb(input_pos_ids.unsqueeze(0).expand_as(input_ids))
        hidden_features = self.dropout(input_features)
        present = []
        for i in range(self.n_layers):
            att_features, cross_present = self.cross_layers[i](hidden_features, ctx_features, ctx_features, ctx_mask, past=past[i][1])
            hidden_features = hidden_features + att_features
            hidden_features, trans_present = self.trans_layers[i](hidden_features, output_mask, past=past[i][0])
            present.append([trans_present, cross_present])

        output_feature = self.output_module(hidden_features)
        return output_feature, present


class GPT2_Head(nn.Module):
    """
    GPT2 head
    """
    def __init__(self, opt):
        super(GPT2_Head, self).__init__()
        self.opt = opt
        self.num_head = getattr(opt, 'num_head', 8)
        self.ctx_layers = 8

        self.act = act()
        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large')
        self.wte = gpt2.transformer.wte
        self.wpe = gpt2.transformer.wpe
        self.dropout = gpt2.transformer.drop
        self.trans_layers = gpt2.transformer.h
        self.n_layers = len(self.trans_layers)
        self.ln_f = gpt2.transformer.ln_f
        self.cross_layers = nn.ModuleList([MultiHeadAttention(opt.common_size, opt.common_size, self.num_head,
                                                              dropout=opt.drop_prob_lm) for _ in range(self.ctx_layers)])
        self.output_module = gpt2.lm_head

        self.txt_position_emb = nn.Parameter(self.init_tensor(80, opt.common_size).unsqueeze(0))
        self.img_position_emb = nn.Parameter(self.init_tensor(80, opt.common_size).unsqueeze(0))
        self.klg_position_emb = nn.Parameter(self.init_tensor(40, opt.common_size).unsqueeze(0))
        self.freeze_parameters()

    def freeze_parameters(self):
        self.wte.weight.requires_grad = False
        self.wpe.weight.requires_grad = False
        for para in self.trans_layers.parameters():
            para.requires_grad = False
        for para in self.ln_f.parameters():
            para.requires_grad = False
        for para in self.output_module.parameters():
            para.requires_grad = False

    def init_tensor(self, dim1, dim2):
        tmp = torch.Tensor(dim1, dim2)
        #nn.init.xavier_uniform_(tmp)
        nn.init.normal_(tmp, std=0.02)
        return tmp

    def forward(self, input_ids, img_feature, txt_features, klg_features, ctx_img_mask=None, ctx_text_mask=None, ctx_know_mask=None, past=None):
        # Get input embeddings
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(1)
        input_word_embs = self.wte(input_ids)


        # Prepare context embeddings
        seq_len = input_word_embs.shape[1]
        output_mask = torch.triu(torch.ones((seq_len, seq_len), device=input_word_embs.device, dtype=input_word_embs.dtype), diagonal=1).unsqueeze(0)

        # Get context masks
        if ctx_img_mask is None and ctx_text_mask is None and ctx_know_mask is None:
            ctx_mask = None
        else:
            if ctx_img_mask is None:
                ctx_img_mask = torch.ones(input_ids.size(0), img_feature.size(1), device=input_ids.device)
            if ctx_text_mask is None:
                ctx_text_mask = torch.ones(input_ids.size(0), txt_features.size(1), device=input_ids.device)
            if ctx_know_mask is None and klg_features is not None:
                ctx_know_mask = torch.ones(input_ids.size(0), klg_features.size(1), device=input_ids.device)

            ctx_mask = 1 - torch.cat((ctx_img_mask, ctx_text_mask, ctx_know_mask), dim=1)
            ctx_mask = ctx_mask.unsqueeze(1).float() * (-1e6)

        if past is not None:
            past_seq_len = past[0][0].shape[3]
            output_mask = torch.cat((torch.zeros((1, seq_len, past_seq_len), device=output_mask.device, dtype=output_mask.dtype), output_mask), dim=-1)

            ctx_features = None
        else:
            past_seq_len = 0
            img_feature = img_feature + self.img_position_emb[:, :img_feature.shape[1]]
            txt_features = txt_features + self.txt_position_emb[:, :txt_features.shape[1]]
            if klg_features is not None:
                klg_features = klg_features + self.klg_position_emb[:, :klg_features.shape[1]]
                ctx_features = torch.cat((img_feature, txt_features, klg_features), dim=1)
            else:
                ctx_features = torch.cat((img_feature, txt_features), dim=1)
            past = [[None, None]] * self.n_layers

        output_mask = output_mask * (-1e6)
        # Input into Transformer layers
        input_pos_ids = torch.arange(past_seq_len, past_seq_len+seq_len, dtype=torch.long, device=input_ids.device)
        input_features = input_word_embs + self.wpe(input_pos_ids.unsqueeze(0).expand_as(input_ids))
        hidden_features = self.dropout(input_features)
        present = []
        for i in range(self.n_layers):
            if i < self.ctx_layers:
                att_features, cross_present = self.cross_layers[i](hidden_features, ctx_features, ctx_features, ctx_mask, past=past[i][1])
                hidden_features = hidden_features + att_features
            else:
                cross_present = None
            hidden_features, trans_present = self.trans_layers[i](hidden_features, attention_mask=output_mask.unsqueeze(1), layer_past=past[i][0])
            present.append([trans_present, cross_present])

        hidden_features = self.ln_f(hidden_features)
        output_feature = self.output_module(hidden_features)
        return output_feature, present
