from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import json
import os

import operator
from tqdm import tqdm

from modules.gpt2.tokenization_gpt2 import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
max_len = 40


splits = ['train', 'val', 'test']


def story_pro(annotations, images):
    print('len_annotations: ', len(annotations))    # 251000
    print('len_image_num: ', len(images))  # 209639
    count = 0
    story = {}

    for split in splits:
        story[split] = []
        for i in tqdm(range(0, len(annotations[split]), 5)):
            try:
                story_id = annotations[split][i]['story_id']

                img_id1, order1 = int(annotations[split][i]["photo_flickr_id"]), annotations[split][i]["worker_arranged_photo_order"]
                img_id2, order2 = int(annotations[split][i + 1]["photo_flickr_id"]), annotations[split][i + 1]["worker_arranged_photo_order"]
                img_id3, order3 = int(annotations[split][i + 2]["photo_flickr_id"]), annotations[split][i + 2]["worker_arranged_photo_order"]
                img_id4, order4 = int(annotations[split][i + 3]["photo_flickr_id"]), annotations[split][i + 3]["worker_arranged_photo_order"]
                img_id5, order5 = int(annotations[split][i + 4]["photo_flickr_id"]), annotations[split][i + 4]["worker_arranged_photo_order"]

                story1 = annotations[split][i]['text']
                story2 = annotations[split][i + 1]['text']
                story3 = annotations[split][i + 2]['text']
                story4 = annotations[split][i + 3]['text']
                story5 = annotations[split][i + 4]['text']
            except json.decoder.JSONDecodeError:
                continue

            img_id_list = [(img_id1, order1), (img_id2, order2), (img_id3, order3), (img_id4, order4),
                           (img_id5, order5)]
            img_id_list = sorted(img_id_list, key=operator.itemgetter(1))
            img_id5 = img_id_list[4][0]

            if not (str(img_id5) in images):
                count += 1
                print('Missing img_id5: ', img_id5)
                print(count)
                continue
            else:
                img_sid5 = str(img_id5)

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

            story_list = [(story1, order1), (story2, order2), (story3, order3), (story4, order4), (story5, order5)]
            story_list = sorted(story_list, key=operator.itemgetter(1))
            story[split].append({'story_id': story_id,
                                 'sent1': story_list[0][0],
                                 'sent2': story_list[1][0],
                                 'sent3': story_list[2][0],
                                 'sent4': story_list[3][0],
                                 'sent5': story_list[4][0],
                                 'last_img_id': img_sid5,
                                 })

        with open('story_{}.json'.format(split), 'w') as ff:
            json.dump(story[split], ff)

    return story


def get_annotation():
    anno = {}
    for split in splits:
        anno[split] = []
        f = json.load(open('%s.story-in-sequence.json' % split))
        annotations = f['annotations']
        print("Number of {} samples: ".format(split), len(annotations))  # 200775 # 24950 # 25275
        for annotation in annotations:
            anno[split].append(annotation[0])

    return anno


def get_images(params):
    images_path_names = os.listdir(params['images_directory'])

    feat_num = []
    print(len(images_path_names))  # 209639
    for i, path in enumerate(images_path_names):
        feat_s = path.split('/')
        feat_num_npz = feat_s[-1]
        feat_num_s = feat_num_npz.split('.')
        feat_num_s = feat_num_s[0]
        feat_num.append(feat_num_s)

    return feat_num


def main(params):
    anno = get_annotation()
    images = get_images(params)
    story = story_pro(anno, images)
    for split in splits:
        print(split, ': ', len(story[split]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_directory', default='ViT_features', help='')
    args = parser.parse_args()
    params = vars(args)

    main(params)
