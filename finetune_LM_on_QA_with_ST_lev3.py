import os
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW

from sklearn.model_selection import train_test_split

from dataset import load_siqa, load_csqa, load_cmqa, load_piqa, prepare_batch
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset
from model import Multiple_Choice_Model
from utils import get_best_model
from utils import Trainer

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur_dir', type=str, default='/home/intern/seungjun/commonsense/CCL')
    parser.add_argument('--root_dir', type=str, default='/home/intern/nas')

    parser.add_argument('--lm', type=str, choices=['roberta-large', 'roberta-cskg'], required=True, help='Pre-trained LM or KG fine-tuned LM.')
    parser.add_argument('--pre_task', type=str, required=True, help='Which QA dataset is used for trainining LM.')
    parser.add_argument('--cur_task', type=str, required=True, choices=['siqa', 'csqa', 'cmqa', 'piqa'], help='Which QA dataset to use for training LM.')
    parser.add_argument('--training_size', type=float, required=True, help='Number of samples to use for training LM.')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--n_epoch', type=int, default=20)
    args = parser.parse_args()
    return args


def main(args):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # dataset and dataloader
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    if args.cur_task == 'siqa':
        target_names = ['Answer A', 'Answer B', 'Answer C']
        train_texts, train_labels = load_siqa(args.root_dir, 'train')
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, train_size=args.training_size, random_state=42)
        train_dataset = SocialiqaDataset(tokenizer, train_texts, train_labels)
        valid_dataset = SocialiqaDataset(tokenizer, valid_texts, valid_labels)
    elif args.cur_task == 'csqa':
        target_names = ['Answer A', 'Answer B', 'Answer C', 'Answer D', 'Answer E']
        train_texts, train_labels = load_csqa(args.root_dir, 'train')
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, train_size=args.training_size, random_state=42)
        train_dataset = CommonsenseqaDataset(tokenizer, train_texts, train_labels)
        valid_dataset = CommonsenseqaDataset(tokenizer, valid_texts, valid_labels)
    elif args.cur_task == 'cmqa':
        target_names = ['Answer A', 'Answer B', 'Answer C', 'Answer D']
        train_texts, train_labels = load_cmqa(args.root_dir, 'train')
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, train_size=args.training_size, random_state=42)
        train_dataset = CosmosqaDataset(tokenizer, train_texts, train_labels)
        valid_dataset = CosmosqaDataset(tokenizer, valid_texts, valid_labels)
    elif args.cur_task == 'piqa':
        target_names = ['Answer A', 'Answer B']
        train_texts, train_labels = load_piqa(args.root_dir, 'train')
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, train_size=args.training_size, random_state=42)
        train_dataset = PhysicaliqaDataset(tokenizer, train_texts, train_labels)
        valid_dataset = PhysicaliqaDataset(tokenizer, valid_texts, valid_labels) 
    print('training_size:', args.training_size)
    print('Train Set:', len(train_dataset), 'Valid Set:', len(valid_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch))


    # define model
    if args.lm == 'roberta-large':
        model_name = args.lm
    elif args.lm == 'roberta-cskg':
        model_name = os.path.join(args.root_dir, args.lm)
    roberta_model = RobertaModel.from_pretrained(model_name)
    model = Multiple_Choice_Model(roberta_model)


    # load trained model
    n_pre_task = int(len(args.pre_task.split('-'))/2)
    if n_pre_task >= 2:
        path = os.path.join(args.root_dir, 'models', 'ST')
        condition = f'{args.lm}-{args.pre_task}'
    else:
        path = os.path.join(args.root_dir, 'models', 'FT')
        condition = f'{args.lm}-{args.pre_task}-bs{args.batch_size}'
    best_name, _ = get_best_model(path, condition)
    best_path = os.path.join(path, best_name)
    restore_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(restore_dict)

    # fine tuning LMs on QA datasets
    save_dir = os.path.join(args.root_dir, 'models', 'ST')
    optimizer = AdamW(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device)
    trainer.run_training(args, train_loader, valid_loader, target_names, save_dir)


if __name__ == '__main__':
    args = parser_args()
    main(args)