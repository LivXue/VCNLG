import json

import numpy as np
import torch
from numpy import mean
from nltk.translate import meteor_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from torchmetrics.functional.multimodal.clip_score import _get_model_and_processor
from tqdm import tqdm


def bleu_scores(gts, res, n=4):
    results = {}
    scorer = Bleu(n)
    score, scores = scorer.compute_score(gts, res)
    if type(score) == list:
        for i, s in enumerate(score):
            results['BLEU {}'.format(i + 1)] = s
            results['BLEU {} scores'.format(i + 1)] = scores[i]
    else:
        results['BLEU 1'] = score
        results['BLEU 1 scores'] = scores

    return results


def cider_scores(gts, res):
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)

    return {'CIDEr': score, 'CIDEr scores': scores}


# This part of code contains some bugs we are unable to fix.
# def meteor_scores(gts, res):
#    scorer = Meteor()
#    score, scores = scorer.compute_score(gts, res)
#
#    return {'METEOR': score, 'METEOR scores':scores}


def meteor_scores(gts, res):
    scores = []
    for id in gts.keys():
        score_m = meteor_score.single_meteor_score(gts[id][0].split(), res[id][0].split())
        scores.append(score_m)
    score = mean(scores)

    return {'METEOR': score, 'METEOR scores': scores}


def rougel_scores(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)

    return {'ROUGE-L': score, 'ROUGE-L scores': scores}


def spice_scores(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)

    return {'SPICE': score, 'SPICE scores': scores}


def clip_scores(res, img_features):
    model, processor = _get_model_and_processor("openai/clip-vit-base-patch32")
    model.eval()
    model = model.cuda()
    b_s = 128
    n_samples = len(res)
    if isinstance(res, dict):
        res = [v[0] for v in res.values()]
    elif isinstance(res, list):
        res = [v[0] for v in res]
    else:
        raise RuntimeError("Invalid res!")

    txt_features = []
    for i in range(0, n_samples, b_s):
        if i + b_s <= n_samples:
            batch = res[i: i + b_s]
        else:
            batch = res[i: n_samples]

        processed_input = processor(text=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            txt_feature = model.get_text_features(
                processed_input["input_ids"][:, :77].cuda(), processed_input["attention_mask"][:, :77].cuda()
            )
        txt_feature = txt_feature / txt_feature.norm(p=2, dim=-1, keepdim=True)
        txt_feature = txt_feature.to(img_features.device)
        txt_features.append(txt_feature)

    txt_features = torch.cat(txt_features, dim=0)
    scores = 100 * (txt_features * img_features).sum(-1)

    score = scores.mean().item()
    score = max(score, 0)
    return {'CLIP': score, 'CLIP scores': scores.cpu()}


def show_all_scores(gts, res, n=4, clip_feature=None):
    """
    :param gts: diction of ground truths
    :param res: diction of references
    :return: Language scores
    """
    if isinstance(gts, list):
        gts = {i: [gts[i]] for i in range(len(gts))}
    if isinstance(res, list):
        res = {i: [res[i]] for i in range(len(res))}
    assert gts.keys() == res.keys(), "ERROR: The keys of references and ground truths are unequal!"

    bleu_results = bleu_scores(gts, res, n=n)
    meteor_results = meteor_scores(gts, res)
    cider_results = cider_scores(gts, res)
    rouge_results = rougel_scores(gts, res)
    rsum = 0.0
    for i in range(n):
        rsum += bleu_results['BLEU {}'.format(i + 1)]
        print("BLEU {} score: {}".format(i + 1, bleu_results['BLEU {}'.format(i + 1)] * 100))
    rsum += meteor_results['METEOR']
    print("METEOR score: {}".format(meteor_results['METEOR'] * 100))
    rsum += cider_results['CIDEr']
    print("CIDEr score: {}".format(cider_results['CIDEr'] * 100))
    rsum += rouge_results['ROUGE-L']
    print("ROUGE-L score: {}".format(rouge_results['ROUGE-L'] * 100))
    print("rSUM score: {}".format(rsum * 100))

    if clip_feature is not None:
        clip_results = clip_scores(res, clip_feature)
        print("CLIP score: {}".format(clip_results['CLIP']))
    else:
        clip_results = None

    return bleu_results, meteor_results, cider_results, rouge_results, rsum, clip_results


def test_stories(end_gen, gold_file="./data/gen/test.txt"):
    with open(gold_file, 'r', encoding='utf-8') as gf:
        gts = [txt.strip('\n') for txt in gf.readlines()]

    res = [txt.lower() for txt in end_gen]
    return show_all_scores(gts, res)


def test_clip(end_gen, dataset="VIST-E"):
    story = json.load(open("data/{}/story_test.json".format(dataset)))
    story = {str(s["story_id"]): s["last_img_id"] for s in story}
    res = json.load(open(end_gen))
    new_res = []
    img_features = []
    for k in tqdm(res.keys()):
        try:
            img_id = story[k]
            img_features.append(np.load("data/{}/clip_features/{}.npy".format(dataset, img_id)))
            new_res.append(res[k])
        except:
            continue

    img_features = torch.tensor(np.stack(img_features))
    score = clip_scores(new_res, img_features)

    return score


if __name__ == '__main__':
    # Input stories
    gold_file = ".results/VIST-E/gts.txt"
    pred_file = "results/VIST-E/res_gpt4_test.txt"

    tgt = "minigpt4_1s.json"
    #score = test_clip("results/VIST-E/" + tgt, dataset="VIST-E")
    #print(score)
    gts = json.load(open("results/VIST-E/gts.json"))
    res = json.load(open("results/VIST-E/" + tgt))
    show_all_scores(gts, res)
