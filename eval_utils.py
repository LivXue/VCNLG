from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import utils
from metrics import show_all_scores


def eval_split(model, loader):
    # Make sure in the evaluation mode
    model.eval()

    gts = loader.dataset.gts
    res = {}
    clip_fea = []

    for data in tqdm(loader):
        data = [_.cuda() for _ in data[:-1]] + [data[-1]]
        ctx_text, ctx_text_mask, ctx_img, ctx_img_mask, ctx_know, ctx_know_mask, output_ids, output_masks, clip_feature, sid = data

        # Forward the model to also get generated samples for each image
        with torch.no_grad():
            seq, seq_logprobs = model(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask, mode='sample')
            seq = seq.cpu()

        # Decode outputs
        for cur_sid, output_ids in zip(sid, seq):
            res[cur_sid] = [loader.dataset.output_tokenizer.decode(output_ids.tolist())]

        clip_fea.append(clip_feature)

    clip_fea = torch.cat(clip_fea, dim=0)
    bleu_results, meteor_results, cider_results, rouge_results, rsum, clip_results = show_all_scores(gts, res, n=4, clip_feature=None)
    lang_stats = {'BLEU 1': bleu_results['BLEU 1'], 'BLEU 2': bleu_results['BLEU 2'],
                  'BLEU 3': bleu_results['BLEU 3'], 'BLEU 4': bleu_results['BLEU 4'],
                  'METEOR': meteor_results['METEOR'], 'CIDEr': cider_results['CIDEr'],
                  'ROUGE-L': rouge_results['ROUGE-L'], 'rSUM': rsum}

    # Switch back to training mode
    model.train()
    return res, lang_stats
