# @FileName  :anomaly.py
# @Time      :2023/10/19 21:25
# @Author    :Duh


import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument('--dataset_dir', type=str, default='data/datasets/')
    parser.add_argument('--feature_dir', type=str, default='data/feats/')
    parser.add_argument('--checkpoint_dir', type=str, default='data/save_models/anomaly/')
    parser.add_argument('--dataset', type=str, default='anomaly')
    parser.add_argument('--mc', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--qmax_words', type=int, default=30)
    parser.add_argument('--amax_words', type=int, default=38)
    parser.add_argument('--max_feats', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_val', type=int, default=128)
    parser.add_argument('--num_thread_reader', type=int, default=8)
    parser.add_argument('--mlm_prob', type=float, default=0)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--embd_dim', type=int, default=512)
    parser.add_argument('--ff_dim', type=int, default=1024)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--freq_display', type=int, default=150)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='data/save_models/anomaly/')

    args = parser.parse_args()

    # Now you can access the command line arguments as attributes of args
    print(args.dataset_dir)
    print(args.feature_dir)
    # ... (access other args in a similar manner)


if __name__ == "__main__":
    conda_env_name = 'mistenv'

    command = "conda run -n mistenv python /home/dh/pythonProject/AnomalyDataset/mist/main_agqa_v2.py --dataset_dir='data/datasets/' --feature_dir='data/feats/' --checkpoint_dir='data/save_models/anomaly/' --dataset=anomaly --mc=0 --epochs=20 --lr=0.00003 --qmax_words=30 --amax_words=38 --max_feats=32 --batch_size=128 --batch_size_val=128 --num_thread_reader=8 --mlm_prob=0 --n_layers=2 --embd_dim=512 --ff_dim=1024 --feature_dim=512 --dropout=0.3 --seed=100 --freq_display=150 --test=0 --save_dir='data/save_models/anomaly/'"
    subprocess.run(command, shell=True)