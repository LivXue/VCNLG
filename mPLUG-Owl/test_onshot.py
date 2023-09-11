from PIL import Image
import json
import os
import random

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from pipeline.interface import get_model
from pipeline.interface import do_generate


if __name__ == '__main__':
    fewshot = 1
    pretrained_ckpt = 'models/mplug-owl-llama-7b'
    model, tokenizer, processor = get_model(pretrained_ckpt=pretrained_ckpt, use_bf16=True)
    model = model.to('cuda:4')

    test_data = json.load(open("../VCNLG/data/VIST-E/story_test.json"))
    train_data = json.load(open("../VCNLG/data/VIST-E/story_train.json"))
    img_root = "../VCNLG/data/VIST-E/images/"
    res = {}

    prompt_prefix = "Please write a continuation within one sentence and within 40 words for the story and according to the ending-related image.\n\n"

    train_temp = []
    train_img = []
    for story in tqdm(train_data):
        cur_sent1 = story['sent1']
        cur_sent2 = story['sent2']
        cur_sent3 = story['sent3']
        cur_sent4 = story['sent4']
        cur_sent5 = story['sent5']
        cur_sid = story['story_id']
        cur_img = story['last_img_id']

        cur_input_txt = ' '.join([cur_sent1, cur_sent2, cur_sent3, cur_sent4])

        if os.path.exists(img_root + cur_img + '.jpg'):
            cur_img = img_root + cur_img + '.jpg'
        elif os.path.exists(img_root + cur_img + '.png'):
            cur_img = img_root + cur_img + '.png'
        else:
            print("Image {} not found!".format(cur_img))
            continue

        train_temp.append("Ending-related image: <image>\n" + "Story: {}\n".format(cur_input_txt) + "Continuation: {}\n\n".format(cur_sent5))
        train_img.append(cur_img)

    train_ids = list(range(len(train_temp)))

    for story in tqdm(test_data):
        cur_sent1 = story['sent1']
        cur_sent2 = story['sent2']
        cur_sent3 = story['sent3']
        cur_sent4 = story['sent4']
        cur_sent5 = story['sent5']
        cur_sid = story['story_id']
        cur_img = story['last_img_id']

        cur_input_txt = ' '.join([cur_sent1, cur_sent2, cur_sent3, cur_sent4])

        if os.path.exists(img_root + cur_img + '.jpg'):
            cur_img = img_root + cur_img + '.jpg'
        elif os.path.exists(img_root + cur_img + '.png'):
            cur_img = img_root + cur_img + '.png'
        else:
            print("Image {} not found!".format(cur_img))
            continue

        tids = random.sample(train_ids, fewshot)
        template = ''.join([train_temp[tid] for tid in tids])
        image_list = [train_img[tid] for tid in tids]

        prompts = [prompt_prefix + template + "Ending-related image: <image>\n" + "Story: {}\n".format(cur_input_txt) + "Continuation: "]
        image_list.append(cur_img)

        continuation = do_generate(prompts, image_list, model, tokenizer, processor,
                                   use_bf16=True, max_length=512, top_k=5, do_sample=True)
        res[cur_sid] = continuation.strip('\n').strip()

    with open('../VCNLG/results/VIST-E/res_mplug_{}shot.json'.format(fewshot), 'w') as ff:
        json.dump(res, ff)
