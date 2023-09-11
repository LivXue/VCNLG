from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import json
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
from utils import opts as opts
from model import VCLM
from dataloader.dataloader import Batch_generator
from eval_utils import eval_split
from modules.loss_wrapper import LossWrapper


def train(opt):
    since = time.time()

    print('...Data loading is beginning...')
    train_set = Batch_generator(opt, mode='train')
    val_set = Batch_generator(opt, mode='val')
    test_set = Batch_generator(opt, mode='test')
    print('...Data loading is completed...')

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, num_workers=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=8, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, num_workers=8, shuffle=False, drop_last=False)
    print("Loaded {} train, {} val, {} test!".format(len(train_set), len(val_set), len(test_set)))

    opt.pad_idx = train_set.pad_id
    opt.bos_idx = train_set.bos_id
    opt.eos_idx = train_set.eos_id
    opt.output_vocab_size = len(train_set.output_tokenizer)
    opt.tokenizer = train_set.output_tokenizer

    model = VCLM(opt).cuda()
    model = model

    lw_model = LossWrapper(model, opt)
    dp_model = torch.nn.DataParallel(model)
    dp_model = dp_model.module  # for single GPU
    dp_lw_model = torch.nn.DataParallel(lw_model)
    dp_lw_model = dp_lw_model.module  # for single GPU
    optimizer = utils.build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), opt)

    dp_lw_model.train()
    early_stop = 0
    best_val_score = 0
    opt.current_lr = opt.learning_rate

    #@torch.compile
    def train_fn(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, know_ids, know_mask, output_ids, output_masks, clip_feature):
        loss = dp_lw_model(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, know_ids, know_mask, output_ids, output_masks, clip_feature)
        loss = loss.mean()
        return loss

    for epoch in range(opt.max_epochs):
        print('Epoch {}/{}'.format(epoch + 1, opt.max_epochs))
        print('-' * 20)
        dp_lw_model.epoch = epoch
        if epoch == opt.reinforce_st_epoch >= 0:
            train_loader = DataLoader(train_set, batch_size=32, num_workers=8, shuffle=True, drop_last=True)
        if epoch >= opt.learning_rate_decay_start >= 0:
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate ** frac
            opt.current_lr = max(opt.learning_rate * decay_factor, 1e-6)
            utils.set_lr(optimizer, opt.current_lr)

        if epoch == opt.finetuning >= 0 and opt.use_bert:
            model.sentence_transformer.fine_tuning = True
            #model.sentence_generator.unfreeze_parameters()

            #@torch.compile
            def train_fn(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, know_ids, know_mask, output_ids, output_masks, clip_feature):
                loss = dp_lw_model(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, know_ids, know_mask, output_ids, output_masks, clip_feature)
                loss = loss.mean()
                return loss

        running_loss = 0.0
        # Load data from train split
        for data in tqdm(train_loader):
            data = [_.cuda() for _ in data[:-1] if _ is not None] + [data[-1]]
            ctx_text, ctx_text_mask, ctx_img, ctx_img_mask, ctx_know, ctx_know_mask, output_ids, output_masks, clip_feature, sid = data

            optimizer.zero_grad()
            loss = train_fn(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask, output_ids, output_masks, clip_feature)
            loss.backward()
            if opt.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' % opt.grad_clip_mode)(model.parameters(), opt.grad_clip_value)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print("Train loss: {}".format(epoch_loss))

        # TODO:
        # Use split=test for easy start and use split=val for strict experiments
        res, lang_stats = eval_split(dp_model, val_loader)

        current_score = lang_stats['rSUM']

        if current_score > best_val_score:
            best_val_score = current_score
            torch.save(model.state_dict(), opt.checkpoint_path + "/best_model.pth")
            print("Saved ", opt.checkpoint_path + "/best_model.pth")
            with open('results/{}/res_{:.4f}.json'.format(opt.dataset, current_score * 100), 'w') as ff:
                json.dump(res, ff)
            print("Saved results/{}/res_{:.4f}.json".format(opt.dataset, current_score * 100))
            early_stop = 0
        else:
            early_stop += 1

        if early_stop == opt.early_stop:
            print("Early stop!")
            break

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model.state_dict(), opt.checkpoint_path + "/epoch{}_model.pth".format(epoch))
    return model


if __name__ == '__main__':
    opt = opts.parse_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    seed = 101
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Random seed = {}".format(seed))

    model = train(opt)
