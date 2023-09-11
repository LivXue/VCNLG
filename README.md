# Vision-Controllable Natural Language Generation

[Dizhan Xue](https://scholar.google.com/citations?user=V5Aeh_oAAAAJ), [Shengsheng Qian](https://scholar.google.com/citations?user=bPX5POgAAAAJ), and [Changsheng Xu](https://scholar.google.com/citations?user=hI9NRDkAAAAJ).

**MAIS, Institute of Automation, Chinese Academy of Sciences**

![](https://img.shields.io/badge/Status-building-brightgreen)

## Examples
  |   |   |
:-------------------------:|:-------------------------:
![example1](figs/exp1.PNG) |  ![example2](figs/exp2.PNG)
![example3](figs/exp3.PNG)  |  ![example4](figs/exp4.PNG)
![example5](figs/exp5.PNG)  |  ![example6](figs/exp6.PNG)
![example7](figs/exp7.PNG)  |  ![example8](figs/exp8.PNG)



## Introduction
- Vision-Controllable Natural Language Generation （VCNLG） aims to continue natural language generation (NLG) following a peceived visual control. 
- Vision-Controllable Language Model (VCLM) aligns a frozen vsiual encoder from BLIP, a frozen textual encoder BERT, and a trained-from-scratch or pretrained generative language model (LM).
- VCLM


![overview](figs/framework.png)


## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/LivXue/VCNLG.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```


**2. Prepare the pretrained LLM weights**

Currently, we provide both Vicuna V0 and Llama 2 version of MiniGPT-4.
Download the corresponding LLM weights from the following huggingface space via clone the repository using git-lfs.

|                                          Vicuna V0 13B                                           |                                          Vicuna V0 7B                                          |                            Llama 2 Chat 7B                             |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Downlad](https://huggingface.co/Vision-CAIR/vicuna/tree/main) | [Download](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) | [Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main)


Then, set the path to the vicuna weight in the model config file 
[here](minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) at Line 18
and/or the path to the llama2 weight in the model config file 
[here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.

**3. Prepare the pretrained MiniGPT-4 checkpoint**

Download the pretrained checkpoints according to the Vicuna model you prepare.

|                                Checkpoint Aligned with Vicuna 13B                                |                                Checkpoint Aligned with Vicuna 7B                                |                            Checkpoint Aligned with Llama 2 Chat 7B                             |
:------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Downlad](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)  | [Download](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing) 


Then, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10) at Line 8 for Vicuna version or [eval_configs/minigpt4_llama2_eval.yaml](eval_configs/minigpt4_llama2_eval.yaml#L10) for LLama2 version.   



### Launching Demo Locally

Try out our demo [demo.py](demo.py) for the vicuna version on your local machine by running

```
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

or for Llama 2 version by 

```
python demo.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
```


To save GPU memory, LLMs loads as 8 bit by default, with a beam search width of 1. 
This configuration requires about 23G GPU memory for 13B LLM and 11.5G GPU memory for 7B LLM. 
For more powerful GPUs, you can run the model
in 16 bit by setting low_resource to False in the config file 
[minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml) and use a larger beam search width.

Thanks [@WangRongsheng](https://github.com/WangRongsheng), you can also run our code on [Colab](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)


### Training
The training of MiniGPT-4 contains two alignment stages.

**1. First pretraining stage**

In the first pretrained stage, the model is trained using image-text pairs from Laion and CC datasets
to align the vision and language model. To download and prepare the datasets, please check 
our [first stage dataset preparation instruction](dataset/README_1_STAGE.md). 
After the first stage, the visual features are mapped and can be understood by the language
model.
To launch the first stage training, run the following command. In our experiments, we use 4 A100. 
You can change the save path in the config file 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage1_pretrain.yaml)

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

A MiniGPT-4 checkpoint with only stage one training can be downloaded 
[here (13B)](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link) or [here (7B)](https://drive.google.com/file/d/1HihQtCEXUyBM1i9DQbaK934wW3TZi-h5/view?usp=share_link).
Compared to the model after stage two, this checkpoint generate incomplete and repeated sentences frequently.


**2. Second finetuning stage**

In the second stage, we use a small high quality image-text pair dataset created by ourselves
and convert it to a conversation format to further align MiniGPT-4.
To download and prepare our second stage dataset, please check our 
[second stage dataset preparation instruction](dataset/README_2_STAGE.md).
To launch the second stage alignment, 
first specify the path to the checkpoint file trained in stage 1 in 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage2_finetune.yaml).
You can also specify the output path there. 
Then, run the following command. In our experiments, we use 1 A100.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

After the second stage alignment, MiniGPT-4 is able to talk about the image coherently and user-friendly. 




## Acknowledgement

+ [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) The model architecture of MiniGPT-4 follows BLIP-2. Don't forget to check this great open-source work if you don't know it before!
+ [Lavis](https://github.com/salesforce/LAVIS) This repository is built upon Lavis!
+ [Vicuna](https://github.com/lm-sys/FastChat) The fantastic language ability of Vicuna with only 13B parameters is just amazing. And it is open-source!


If you're using MiniGPT-4 in your research or applications, please cite using this BibTeX:
```bibtex
@article{zhu2023minigpt,
  title={MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models},
  author={Zhu, Deyao and Chen, Jun and Shen, Xiaoqian and Li, Xiang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2304.10592},
  year={2023}
}
```


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).
