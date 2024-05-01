import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import logging,argparse
from transformers import get_cosine_schedule_with_warmup, DistilBertTokenizer, BertTokenizer
from args import get_args
from model.mist import MIST_VideoQA
from util import compute_a2v
from train.train_anomaly import train, eval
from data.agqa_clip_patch_loader import get_videoqa_loaders




if __name__ == "__main__":
    # args, logging
    args = get_args()
    if not (os.path.isdir(args.save_dir)):
        os.mkdir(os.path.join(args.save_dir))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "stdout.log"), "w+")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    logging.info(args)

    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # get answer embeddings
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    a2id, id2a, a2v = None, None, None
    if not args.mc:
        a2id, id2a, a2v = compute_a2v(
            vocab_path=args.vocab_path,
            bert_tokenizer=bert_tokenizer,
            amax_words=args.amax_words,
        )
        logging.info(f"Length of Answer Vocabulary: {len(a2id)}")

    # Model
    model = MIST_VideoQA(
        feature_dim=args.feature_dim,
        word_dim=args.word_dim,
        N=args.n_layers,
        d_model=args.embd_dim,
        d_ff=args.ff_dim,
        h=args.n_heads,
        dropout=args.dropout,
        T=args.max_feats,
        Q=args.qmax_words,
        baseline=args.baseline,
    )
    model.cuda()
    logging.info("Using {} GPUs".format(torch.cuda.device_count()))

    # Load pretrain path
    model = nn.DataParallel(model)
    if args.pretrain_path != "":
        model.load_state_dict(torch.load(args.pretrain_path))
        logging.info(f"Loaded checkpoint {args.pretrain_path}")
    logging.info(
        f"Nb of trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Dataloaders
    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_videoqa_loaders(args, args.features_path, a2id, bert_tokenizer, test_mode=args.test)

    if args.test:
        logging.info("number of test instances: {}".format(len(test_loader.dataset)))
    else:
        logging.info("number of train instances: {}".format(len(train_loader.dataset)))
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))


    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    params_for_optimization = list(p for n, p in model.named_parameters() if p.requires_grad and n.split('.')[1] != 'clip')
    optimizer = optim.Adam(
        params_for_optimization, lr=args.lr, weight_decay=args.weight_decay
    )
    criterion.cuda()

    # Training
    if not args.test:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 0, len(train_loader) * args.epochs
        )
        logging.info(
            f"Set cosine schedule with {len(train_loader) * args.epochs} iterations"
        )
        print(len(train_loader))
        best_val_acc = -float("inf")
        best_epoch = 0
        for epoch in range(args.epochs):
            best_val_acc, best_epoch = train(model, train_loader, a2v, optimizer, criterion, scheduler, epoch, args,
                                             val_loader, best_val_acc, best_epoch)
        logging.info(f"Best val model at epoch {best_epoch + 1}")
#        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model_1020.pth"))

    # Evaluate on test set
    eval(model, test_loader, a2v, args, test=True)
