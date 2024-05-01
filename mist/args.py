import argparse
import os

from global_parameters import (
    DEFAULT_DATASET_DIR,
    DEFAULT_CKPT_DIR,
    TRANSFORMERS_PATH,
    SSD_DIR
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='anomaly',
    )

    # Model
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        choices=["", "qa"],
        help="qa baseline does not use the video, video baseline does not use the question",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="number of layers in the multi-modal transformer",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="number of attention heads in the multi-modal transformer",
    )
    parser.add_argument(
        "--embd_dim",
        type=int,
        default=512,
        help="multi-modal transformer and final embedding dimension",
    )
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=2048,
        help="multi-modal transformer feed-forward dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout rate in the multi-modal transformer",
    )
    parser.add_argument(
        "--sentence_dim",
        type=int,
        default=2048,
        help="sentence dimension for the differentiable bag-of-words embedding the answers",
    )
    parser.add_argument(
        "--qmax_words",
        type=int,
        default=20,
        help="maximum number of words in the question",
    )
    parser.add_argument(
        "--amax_words",
        type=int,
        default=50,
        help="maximum number of words in the answer",
    )
    parser.add_argument(
        "--max_feats",
        type=int,
        default=20,
        help="maximum number of video features considered",
    )

    # Paths
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default='../data/datasets/anomaly/',
        help="folder where the datasets folders are stored",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default='../data/feats',
        help="folder where the datasets folders are stored",
    )
    parser.add_argument(
        "--checkpoint_predir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="folder to store checkpoints",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default='data/save_models/anomaly/', help="subfolder to store checkpoint"
    )
    parser.add_argument(
        "--pretrain_path", type=str, default="data/save_models/anomaly/best_model_1020.pth", help="path to pretrained checkpoint"
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        default=TRANSFORMERS_PATH,
        help="path to transformer models checkpoints",
    )


    # Train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_val", type=int, default=8)
    parser.add_argument(
        "--n_pair",
        type=int,
        default=32,
        help="number of clips per video to consider to train on HowToVQA69M",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--test", type=int, default=0, help="use to evaluate without training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.00005, help="initial learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--clip",
        type=float,
        default=12,
        help="gradient clipping",
    )

    # Print
    parser.add_argument(
        "--freq_display", type=int, default=5, help="number of train prints per epoch"
    )
    parser.add_argument(
        "--num_thread_reader", type=int, default=16, help="number of workers"
    )

    # Masked Language Modeling and Cross-Modal Matching parameters
    parser.add_argument("--mlm_prob", type=float, default=0)
    parser.add_argument("--n_negs", type=int, default=1)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--min_time", type=int, default=10)
    parser.add_argument("--min_words", type=int, default=10)

    # Demo parameters
    parser.add_argument(
        "--question_example", type=str, default="", help="demo question text"
    )
    parser.add_argument("--video_example", type=str, default="", help="demo video path")
    parser.add_argument("--port", type=int, default=8899, help="demo port")
    parser.add_argument(
        "--pretrain_path2", type=str, default="", help="second demo model"
    )
    parser.add_argument(
        "--save_dir", type=str, default="data/save_models/anomaly/", help="path to save dir"
    )
    parser.add_argument(
        "--mc", type=int, default=5, help="number of multiple choices"
    )
    parser.add_argument(
        "--bnum", type=int, default=10, help="number of region proposal"
    )
    parser.add_argument(
        "--feature_dim", type=int, default=512, help="whether to finetune the weights pretrained on WebVid"
    )

    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = args.bert_path

    # feature dimension
    args.word_dim = 768  # DistilBERT

    load_path = os.path.join(args.dataset_dir, args.dataset)
    args.load_path = load_path

    args.features_path = os.path.join(args.feature_dir, args.dataset)

    args.train_csv_path = os.path.join(load_path, "train.csv")
    args.val_csv_path = os.path.join(load_path, "val.csv")
    args.test_csv_path = os.path.join(load_path, "test.csv")
    args.vocab_path = os.path.join(load_path, "vocab.json")

    return args
