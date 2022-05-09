import os
import json
import random
import pathlib
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import RobertaTokenizer, RobertaModel, AdamW

from sklearn.model_selection import train_test_split

from dataset import biencoder_batch, load_siqa_answersheet, load_csqa_answersheet, load_cmqa_answersheet, load_piqa_answersheet, BiEncoderDataset
from model import BiEncoder
from utils import make_dir, TrainerForEncoder

import warnings
warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur_task', type=str, required=True, choices=['siqa', 'csqa', 'cmqa', 'piqa'], help='Which QA dataset to use for training LM.')
    parser.add_argument('--training_size', type=int, required=True, help='Number of samples to use for training LM.')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--n_epoch', type=int, default=20)
    args = parser.parse_args()
    return args


def main(args):
    roberta_model_1 = RobertaModel.from_pretrained('roberta-large')
    roberta_model_2 = RobertaModel.from_pretrained('roberta-large')
    root_dir = '/home/intern/nas'
    base_dir = os.path.join('/home/intern/nas/CCL/bi-encoder', args.cur_task + f'-ts{args.training_size}')
    make_dir(base_dir)
    print('Directory:', base_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    if args.cur_task == 'siqa':
        train_data = load_siqa_answersheet(root_dir)
    elif args.cur_task =='cmqa':
        train_data = load_cmqa_answersheet(root_dir)
    elif args.cur_task == 'csqa':
        train_data = load_csqa_answersheet(root_dir)
    elif args.cur_task == 'piqa':
        train_data = load_piqa_answersheet(root_dir)
    train_data, valid_data = train_test_split(train_data, train_size=args.training_size, random_state=42)
    train_dataset = BiEncoderDataset(tokenizer, train_data)
    valid_dataset = BiEncoderDataset(tokenizer, valid_data)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: biencoder_batch(batch))
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: biencoder_batch(batch))
    model = BiEncoder(roberta_model_1, roberta_model_2)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    trainer = TrainerForEncoder(model, optimizer, device)
    trainer.run_training(args, train_loader, valid_loader, base_dir)

if __name__ == '__main__':
    args = parser_args()
    main(args)