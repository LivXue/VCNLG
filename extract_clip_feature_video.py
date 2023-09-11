import os
import argparse

import torch
from torchmetrics.multimodal import CLIPScore
from torchmetrics.functional.multimodal.clip_score import _get_model_and_processor
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np


def load_image(image):
    image = Image.open(str(image)).convert('RGB')

    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #])
    #image = transform(image).unsqueeze(0)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/data/LSMDC-E/videos')
    parser.add_argument('--output_dir', default='data/LSMDC-E/clip_features')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    model, processor = _get_model_and_processor("openai/clip-vit-base-patch32")
    model = model.to(args.device)
    video_list = os.listdir(args.input_dir)

    for video in tqdm(video_list):
        video_path = os.path.join(args.input_dir, video)
        frame_list = os.listdir(video_path)
        clip_feas = []
        for img in frame_list:
            im = load_image(os.path.join(video_path, img))
            processed_input = processor(images=[im], return_tensors="pt")
            with torch.no_grad():
                img_features = model.get_image_features(processed_input["pixel_values"].to(args.device))
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
            img_features = img_features.squeeze().cpu()
            clip_feas.append(img_features.unsqueeze(0))

        np.save(args.output_dir + "/{}.npy".format(video), torch.cat(clip_feas, dim=0).mean(dim=0).numpy())

