import os
import argparse
import time

from PIL import Image
import numpy as np
import yaml
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import VisionTransformer
from models.blip_pretrain import blip_pretrain



def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')

    #w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='<your root>/VCNLG/data/VIST-E/images', help='path to image file folder')
    parser.add_argument('--output_dir', default='<your root>/VCNLG/data/VIST-E/ViT_features', help='path to store ViT features')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    config = yaml.load(open("./configs/pretrain.yaml", 'r'), Loader=yaml.Loader)
    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                          vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
    model.load_state_dict(torch.load("./checkpoints/model_large.pth")['model'])
    model = model.visual_encoder.to(args.device)
    model.eval()

    img_list = os.listdir(args.input_dir)
    cost_time = 0
    for i, img in tqdm(enumerate(img_list)):
        try:
            im = load_image(os.path.join(args.input_dir, img), 224, args.device)
        except:
            print("Cannot open", img)
            continue
        with torch.no_grad():
            since = time.time()
            vit_fea = model(im).squeeze()
            cost_time += time.time() - since
        print(cost_time/ (i+1))
        vit_fea = vit_fea.cpu().numpy().astype(np.float32)

        np.save(os.path.join(args.output_dir, img.split('.')[0] + '.npy'), vit_fea)
