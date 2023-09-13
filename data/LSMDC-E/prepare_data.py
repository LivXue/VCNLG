import re
import json
import argparse
import os
import csv

import numpy as np
from tqdm import tqdm

from modules.gpt2.tokenization_gpt2 import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')


splits = ['train', 'val', 'test']
max_len = 40


def collect_story(params):
    # count up the number of words
    max_len = []
    csvs = ['LSMDC16_annos_training_someone.csv', 'LSMDC16_annos_val_someone.csv', 'LSMDC16_annos_test_someone.csv']
    for c in csvs[:-1]:
        with open(os.path.join(params['input_path'], c)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                # remove punctuation but keep possessive because we want to separate out character names
                ws = str(row[5]).split()
                max_len.append(len(ws))
    print('mean setence length', np.mean(max_len))

    # lets now produce the final annotations
    videos = []
    movie_ids = {}
    vid = 0
    groups = []
    gid = -1
    for i, c in enumerate(csvs):
        split = splits[i]
        with open(os.path.join(params['input_path'], c)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                clip = row[0]
                movie = clip[:clip.rfind('_')]
                info = {'id': vid, 'split': split, 'movie': movie, 'clip': clip}
                if movie not in movie_ids:
                    gid += 1
                    ginfo = {'id': gid, 'split': split, 'movie': movie, 'videos': [vid]}
                    groups.append(ginfo)
                    gcount = 0
                    movie_ids[movie] = [gid]
                else:
                    if gcount >= params['group_by']:
                        gid += 1
                        ginfo = {'id': gid, 'split': split, 'movie': movie, 'videos': [vid]}
                        groups.append(ginfo)
                        gcount = 0
                        movie_ids[movie].append(gid)
                    else:
                        groups[gid]['videos'].append(vid)
                if split != 'blind_test':
                    info['final_caption'] = str(row[5])
                videos.append(info)
                vid += 1
                gcount += 1
    return videos, groups, movie_ids


def story_pro(videos, groups):
    story = {'train': [], 'val': [], 'test': []}
    count = 0
    for ginfo in tqdm(groups):
        if len(ginfo['videos']) != 5:
            continue
        story1 = process_gpt_tokens(videos[ginfo['videos'][0]]['final_caption'])
        story2 = process_gpt_tokens(videos[ginfo['videos'][1]]['final_caption'])
        story3 = process_gpt_tokens(videos[ginfo['videos'][2]]['final_caption'])
        story4 = process_gpt_tokens(videos[ginfo['videos'][3]]['final_caption'])
        story5 = process_gpt_tokens(videos[ginfo['videos'][4]]['final_caption'])

        story1_token = tokenizer.encode(story1)
        story2_token = tokenizer.encode(story2)
        story3_token = tokenizer.encode(story3)
        story4_token = tokenizer.encode(story4)
        story5_token = tokenizer.encode(story5)

        if len(story1_token) > max_len:
            count += 1
            print(count)
            continue
        if len(story2_token) > max_len:
            count += 1
            print(count)
            continue
        if len(story3_token) > max_len:
            count += 1
            print(count)
            continue
        if len(story4_token) > max_len:
            count += 1
            print(count)
            continue
        if len(story5_token) > max_len:
            count += 1
            print(count)
            continue

        last_seg_id = videos[ginfo['videos'][4]]['clip']
        split = videos[ginfo['videos'][4]]['split']


        story[split].append({'story_id': ginfo['id'],
                             'sent1': story1,
                             'sent2': story2,
                             'sent3': story3,
                             'sent4': story4,
                             'sent5': story5,
                             'last_img_id': last_seg_id,
                             })

    for split in splits:
        with open('story_{}.json'.format(split), 'w') as ff:
            json.dump(story[split], ff)

    return story


def process_gpt_tokens(s):
    v = tokenizer.encode(s)
    # Add blanks before [,.?!]
    for i in range(len(v)):
        if v[i] == 11:
            v[i] = 837
        elif v[i] == 13:
            v[i] = 764
        elif v[i] == 30:
            v[i] = 5633
        elif v[i] == 0:
            v[i] = 5145

    v = tokenizer.decode(v)
    v = v.lower()
    v = v.replace("n't", " n't").replace("'m", " 'm").replace("'s", " 's").replace("'ll", " 'll").replace("s' ", "s ' ")
    v = v.strip()
    return v


def main(params):
    # create the vocab
    videos, groups, movie_ids = collect_story(params)
    story = story_pro(videos, groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_path', type=str, default='./',
                        help='directory containing csv files')
    parser.add_argument('--group_by', default=5, type=int,
                        help='group # of clips as one video')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
