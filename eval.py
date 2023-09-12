from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os

import torch
import json
from torch.utils.data import DataLoader
import utils
import utils.opts as opts
from model import VCLM
from eval_utils import eval_split
from dataloader.dataloader import Batch_generator


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    opt = opts.parse_opt()
    # Input paths
    opt.model = './log/{}/log_5/best_model.pth'.format(opt.dataset)

    test_set = Batch_generator(opt, mode='test')
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, num_workers=3, shuffle=False, drop_last=False)
    opt.pad_idx = test_set.pad_id
    opt.bos_idx = test_set.bos_id
    opt.eos_idx = test_set.eos_id
    opt.output_vocab_size = len(test_set.output_tokenizer)
    opt.tokenizer = test_set.output_tokenizer

    # Setup the model
    model = VCLM(opt)
    model.load_state_dict(torch.load(opt.model))
    model = model.cuda()

    # Set sample options
    res, lang_stats = eval_split(model, test_loader, clip=True)
    current_score = lang_stats['rSUM']

    #with open('results/{}/res_{:.4f}.json'.format(opt.dataset, current_score * 100), 'w') as ff:
    #    json.dump(res, ff)
    if lang_stats:
        print(lang_stats)
