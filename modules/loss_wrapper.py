import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from torchmetrics.functional.multimodal.clip_score import _get_model_and_processor

import model
import modules.losses as losses


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = losses.AdvancedLMC()
        self.clip_crit = losses.AdvancedLMC('none')
        self.kl_loss = KLDivLoss(reduction="batchmean", log_target=True)
        self.clip_model, self.clip_processor = _get_model_and_processor("openai/clip-vit-base-patch32")
        self.clip_model = self.clip_model.cuda()
        self.clip_model.eval()
        self.epoch = 0

    def forward(self, ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask, output_ids, output_masks,
                clip_feature):
        outputs, reuse_feature = self.model(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask,
                                            output_ids[..., :-1], return_feature=True)
        loss = self.crit(outputs, output_ids[..., 1:], output_masks[..., 1:])
        if self.epoch >= self.opt.reinforce_st_epoch >= 0:
            reinforce_loss = self.vision_reinforce(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know,
                                                   ctx_know_mask, clip_feature, reuse_feature)
            return loss + 0.1 * reinforce_loss
        else:
            return loss

    def vision_reinforce(self, ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask, clip_imgs,
                         reuse_feature=None, n_sampling=4):
        losses, rewards, logits = [], [], []
        for _ in range(n_sampling):
            res = []
            self.model.eval()
            with torch.no_grad():
                if reuse_feature is not None:
                    seq, seqLogits = self.model(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask,
                                                greedy=False, reuse_features=reuse_feature, mode='sample')
                else:
                    seq, seqLogits, reuse_feature = self.model(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know,
                                                               ctx_know_mask, greedy=False, return_feature=True,
                                                               mode='sample')
            mask = (seq != self.opt.pad_idx)
            mask[(torch.arange(mask.size(0)), mask.long().sum(dim=-1).clamp(max=mask.size(1) - 1))] = True
            input_seq = torch.cat(
                (torch.full((seq.size(0), 1), fill_value=self.opt.bos_idx, dtype=torch.long, device=seq.device), seq),
                dim=1)

            self.model.train()
            outputs = self.model(ctx_img, ctx_img_mask, ctx_text, ctx_text_mask, ctx_know, ctx_know_mask,
                                 input_seq[..., :-1], reuse_features=reuse_feature)
            loss = self.clip_crit(outputs, seq, mask.long())
            loss = loss.view(seq.size(0), seq.size(1)).masked_fill(~mask, 0).sum(
                -1)  # / mask.float().sum(-1).clamp(min=1)
            losses.append(loss.unsqueeze(0))
            logits.append(seqLogits.unsqueeze(0))

            for output_ids in seq:
                res.append(self.opt.tokenizer.decode(output_ids.cpu().tolist()))

            processed_input = self.clip_processor(text=res, return_tensors="pt", padding=True)
            with torch.no_grad():
                txt_feature = self.clip_model.get_text_features(
                    processed_input["input_ids"][:, :77].to(ctx_img.device),
                    processed_input["attention_mask"][:, :77].to(ctx_img.device)
                )
            txt_feature = txt_feature / txt_feature.norm(p=2, dim=-1, keepdim=True)
            clip_scores = 100 * (txt_feature * clip_imgs).sum(-1)
            rewards.append(clip_scores.unsqueeze(0))

        # Normalize rewards
        rewards = torch.cat(rewards, dim=0)
        losses = torch.cat(losses, dim=0)
        # r_mean = rewards.mean(dim=0, keepdim=True)
        # rewards = (rewards - r_mean) / 100
        reinforce_loss = (losses * rewards / 100).mean()
        # reinforce_loss = - (torch.exp(-losses + losses.detach()) * rewards).mean()
        kl_loss = 0
        for seqLogits in logits:
            log_p = F.log_softmax(seqLogits, dim=-1)
            kl_loss = kl_loss + self.kl_loss(log_p, log_p.detach())
        kl_loss = kl_loss / len(logits)

        return reinforce_loss + kl_loss
