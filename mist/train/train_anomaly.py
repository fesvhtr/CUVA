# @FileName  :train_anomaly.py
# @Time      :2023/10/7 21:49
# @Author    :Duh
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
import os
import json
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import h5py
from tqdm import tqdm
from IPython.core.debugger import Pdb

dbg = Pdb()


def eval(model, val_loader, a2v, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)
    results = {}
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            video_id, video, question, question_clip, answer_emb, labels = (
                batch["video_id"],
                (batch["video"][0].cuda(), batch["video"][1].cuda()),
                batch["question"].cuda(),
                batch['question_clip'].cuda(),
                batch["answer_emb"].cuda(),
                batch["label"]
            )
            video_len = batch["video_len"]
            question_mask = (question > 0).float()
            video_mask = get_mask(video_len, video[1].size(1)).cuda()
            count += labels.size(0)
            fusion_proj, answer_proj = model(
                video,
                question,
                text_mask=question_mask,
                # video_mask=video_mask,
                answer=answer_emb,
                question_clip=question_clip
            )
            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

            predicted = torch.max(predicts, dim=-1).indices.cpu().unsqueeze(-1)

            # pre = torch.zeros_like(labels).scatter_(1, predicted, 1)

            labels = torch.max(labels, dim=1).indices.cpu().unsqueeze(-1)
            metrics["acc"] += (predicted == labels).sum().item()
            for bs, video_id in enumerate(batch['video_id']):
                results[video_id] = {'prediction': int(predicted.numpy()[bs])}

    step = "val" if not test else "test"
    for k in metrics:
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
    acc = metrics['acc'] / count
    json.dump(results, open(os.path.join(args.save_dir, f"val-{acc:.5%}.json"), "w"))

    return metrics["acc"] / count


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, val_loader=None, best_val_acc=None,
          best_epoch=None):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    for i, batch in enumerate(train_loader):
        video, question, question_clip, answer_emb, labels = (
            (batch["video"][0].cuda(), batch["video"][1].cuda()),
            batch["question"].cuda(),
            batch['question_clip'].cuda(),
            batch["answer_emb"].cuda(),
            batch["label"].cuda()
        )
        video_len = batch["video_len"]
        question_mask = (question > 0).float()
        # video_mask = (
        #     get_mask(video_len, video[1].size(1)).cuda() if args.max_feats > 0 else None
        # )
        N = labels.size(0)
        fusion_proj, answer_proj = model(
            video,
            question,
            text_mask=question_mask,
            # video_mask=video_mask,
            answer=answer_emb,
            question_clip=question_clip
        )
        fusion_proj = fusion_proj.unsqueeze(2)  # (bs, nans, 768)
        predicts = torch.bmm(answer_proj, fusion_proj).squeeze().cpu()
        labels = torch.max(labels, dim=1).indices.cpu().squeeze()
        vqa_loss = criterion(predicts, labels)
        predicted = torch.max(predicts, dim=1).indices.unsqueeze(-1)
        running_acc.update((predicted == labels.unsqueeze(-1)).sum().item() / N, N)

        loss = vqa_loss

        if torch.isnan(loss):
            print(batch['question_id'], batch['video_id'], loss)
            dbg.set_trace()
        # dbg.set_trace()
        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)

        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, Training MLM loss: {running_mlm_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}, Epoch status: {float(i + 1) / len(train_loader):.4f}, Training VideoQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}"
                )
            running_acc.reset()
            running_vqa_loss.reset()

            running_mlm_loss.reset()

        if val_loader is not None and ((epoch + 1) % args.freq_display) == 0:
            val_acc = eval(model, val_loader, a2v, args, test=False)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
                print("best_epoch %d save" % best_epoch)
            # else:
            #     torch.save(
            #         model.state_dict(), os.path.join(args.save_dir, f"model-{epoch}.pth")
            #     )

    return best_val_acc, best_epoch


def infer(model, infer_loader, args, h5pyfile):
    model.eval()
    dataset_feats = h5py.File(h5pyfile, "w")
    dataset_feats.create_dataset("features", (len(infer_loader.dataset), 512))  # len(val_loader.dataset)
    dataset_feats.create_dataset("ids", (len(infer_loader.dataset),), 'S20')
    globalidx = 0
    with torch.no_grad():
        for batch in tqdm(infer_loader):
            video_id, video, question, question_clip, answer_emb, labels = (
                batch["video_id"],
                (batch["video"][0].cuda(), batch["video"][1].cuda()),
                batch["question"].cuda(),
                batch['question_clip'].cuda(),
                batch["answer_emb"].cuda(),
                batch["label"]
            )
            batch_size = len(batch["video_id"])
            question_mask = (question > 0).float()
            fusion_proj, answer_proj = model(
                video,
                question,
                text_mask=question_mask,
                answer=answer_emb,
                question_clip=question_clip
            )
            for j in range(batch_size):
                dataset_feats['ids'][globalidx] = batch["video_id"][j].encode("ascii", "ignore")
                # print(dataset_feats['ids'][globalidx])
                dataset_feats['features'][globalidx] = fusion_proj.cpu()[j]
                # print(dataset_feats['features'][globalidx])
                globalidx += 1

    return globalidx, h5pyfile
    #return "{} videos features extracted done in {}".format(globalidx, h5pyfile)