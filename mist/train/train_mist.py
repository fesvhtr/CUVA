import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
import os
import json
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens

from IPython.core.debugger import Pdb
dbg = Pdb()


def eval(model, val_loader, a2v, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)
    results = {}
    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        for i, batch in enumerate(val_loader):
            answer_id, answer, video, question, question_clip = (
                batch["answer_id"],
                batch["answer"],
                (batch["video"][0].cuda(), batch["video"][1].cuda()),
                batch["question"].cuda(),
                batch['question_clip'].cuda()
            )
            video_len = batch["video_len"]
            question_mask = (question > 0).float()
            video_mask = get_mask(video_len, video[1].size(1)).cuda()
            count += answer_id.size(0)

            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    # video_mask=video_mask,
                    question_clip=question_clip
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(batch['question_id']):
                    results[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}
                    
            else:
                fusion_proj, answer_proj = model(
                    video,
                    question,
                    text_mask=question_mask,
                    # video_mask=video_mask,
                    answer=answer.cuda(),
                    question_clip=question_clip
                )
                fusion_proj = fusion_proj.unsqueeze(2)
                # todo: calculate the sim
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                print(predicts)
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(batch['question_id']):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}

    step = "val" if not test else "test"
    for k in metrics:
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
    acc = metrics['acc'] / count
    json.dump(results, open(os.path.join(args.save_dir, f"val-{acc:.5%}.json"), "w"))

    return metrics["acc"] / count


def train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args, val_loader=None, best_val_acc=None, best_epoch=None):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    for i, batch in enumerate(train_loader):
        answer_id, answer, video, question, question_clip = (
            batch["answer_id"],
            batch["answer"],
            (batch["video"][0].cuda(), batch["video"][1].cuda()),
            batch["question"].cuda(),
            batch['question_clip'].cuda()
        )
        video_len = batch["video_len"]
        question_mask = (question > 0).float()
        # video_mask = (
        #     get_mask(video_len, video[1].size(1)).cuda() if args.max_feats > 0 else None
        # )
        N = answer_id.size(0)
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            predicts = model(
                video,
                question,
                text_mask=question_mask,
                # video_mask=video_mask,
                question_clip=question_clip
            )
        else:
            fusion_proj, answer_proj = model(
                video,
                question,
                text_mask=question_mask,
                # video_mask=video_mask,
                answer=answer.cuda(),
                question_clip=question_clip
            )
            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

        if args.dataset == "ivqa":
            a = (answer_id / 2).clamp(max=1).cuda()
            vqa_loss = criterion(predicts, a)
            predicted = torch.max(predicts, dim=1).indices.cpu()
            predicted = F.one_hot(predicted, num_classes=len(a2v))
            running_acc.update((predicted * a.cpu()).sum().item() / N, N)
        else:
            vqa_loss = criterion(predicts, answer_id.cuda())
            predicted = torch.max(predicts, dim=1).indices.cpu()
            running_acc.update((predicted == answer_id).sum().item() / N, N)

        if args.mlm_prob:
            inputs = batch["question"]
            inputs, labels = mask_tokens(
                inputs, model.module.bert.bert_tokenizer, mlm_probability=0.15
            )
            mlm_loss = model(
                video,
                question=inputs.cuda(),
                labels=labels.cuda(),
                text_mask=question_mask,
                video_mask=video_mask,
                mode="mlm",
            )
            mlm_loss = mlm_loss.mean()
            loss = mlm_loss + vqa_loss
        else:
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
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)

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


        if val_loader is not None and (i + 1) % (len(train_loader) // (args.freq_display / 15)) == 0:
            val_acc = eval(model, val_loader, a2v, args, test=False)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
                )
            # else:
            #     torch.save(
            #         model.state_dict(), os.path.join(args.save_dir, f"model-{epoch}.pth")
            #     )

    return best_val_acc, best_epoch
