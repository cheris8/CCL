import os
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, AdamW

from sklearn.model_selection import train_test_split

from dataset import load_siqa, load_csqa, load_cmqa, load_piqa, prepare_batch_KD
from dataset import SocialiqaDatasetForKD, CommonsenseqaDatasetForKD, CosmosqaDatasetForKD, PhysicaliqaDatasetForKD
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset
from model import Multiple_Choice_Model
from utils import get_predictions_from_pre_model, get_best_model
from utils import CriterionForKD, TrainerForKD


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur_dir', type=str, default='/home/chaehyeong/CKL')
    parser.add_argument('--root_dir', type=str, default='/home/chaehyeong/nas')

    parser.add_argument('--lm', type=str, default='roberta-large', choices=['roberta-large', 'roberta-cskg'], help='Pre-trained LM or KG fine-tuned LM.')
    parser.add_argument('--pre_task', type=str, required=True, help='Which QA dataset is used for trainining LM.')
    parser.add_argument('--cur_task', type=str, required=True, choices=['siqa', 'csqa', 'cmqa', 'piqa'], help='Which QA dataset to use for training LM.')
    parser.add_argument('--training_size', type=int, required=True, help='Number of samples to use for training LM.')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--T', type=float, default=1)
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset and dataloader
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    if args.cur_task == 'siqa':
        target_names = ['Answer A', 'Answer B', 'Answer C']
        train_texts, train_labels = load_siqa(args.root_dir, 'train')
        train_pseudo_labels = get_predictions_from_pre_model(args, SocialiqaDataset(tokenizer, train_texts, train_labels), device)
        train_texts, valid_texts, train_labels, valid_labels, train_pseudo_labels, valid_pseudo_labels \
            = train_test_split(train_texts, train_labels, train_pseudo_labels, train_size=args.training_size, random_state=42)
        train_dataset = SocialiqaDatasetForKD(tokenizer, train_texts, train_labels, train_pseudo_labels)
        valid_dataset = SocialiqaDatasetForKD(tokenizer, valid_texts, valid_labels, valid_pseudo_labels)
    elif args.cur_task == 'csqa':
        target_names = ['Answer A', 'Answer B', 'Answer C', 'Answer D', 'Answer E']
        train_texts, train_labels = load_csqa(args.root_dir, 'train')
        train_pseudo_labels = get_predictions_from_pre_model(args, CommonsenseqaDataset(tokenizer, train_texts, train_labels), device)
        train_texts, valid_texts, train_labels, valid_labels, train_pseudo_labels, valid_pseudo_labels \
            = train_test_split(train_texts, train_labels, train_pseudo_labels, train_size=args.training_size, random_state=42)
        train_dataset = CommonsenseqaDatasetForKD(tokenizer, train_texts, train_labels, train_pseudo_labels)
        valid_dataset = CommonsenseqaDatasetForKD(tokenizer, valid_texts, valid_labels, valid_pseudo_labels)
    elif args.cur_task == 'cmqa':
        target_names = ['Answer A', 'Answer B', 'Answer C', 'Answer D']
        train_texts, train_labels = load_cmqa(args.root_dir, 'train')
        train_pseudo_labels = get_predictions_from_pre_model(args, CosmosqaDataset(tokenizer, train_texts, train_labels), device)
        train_texts, valid_texts, train_labels, valid_labels, train_pseudo_labels, valid_pseudo_labels \
            = train_test_split(train_texts, train_labels, train_pseudo_labels, train_size=args.training_size, random_state=42)
        train_dataset = CosmosqaDatasetForKD(tokenizer, train_texts, train_labels, train_pseudo_labels)
        valid_dataset = CosmosqaDatasetForKD(tokenizer, valid_texts, valid_labels, valid_pseudo_labels)
    elif args.cur_task == 'piqa':
        target_names = ['Answer A', 'Answer B']
        train_texts, train_labels = load_piqa(args.root_dir, 'train')
        train_pseudo_labels = get_predictions_from_pre_model(args, PhysicaliqaDataset(tokenizer, train_texts, train_labels), device)
        train_texts, valid_texts, train_labels, valid_labels, train_pseudo_labels, valid_pseudo_labels \
            = train_test_split(train_texts, train_labels, train_pseudo_labels, train_size=args.training_size, random_state=42)
        train_dataset = PhysicaliqaDatasetForKD(tokenizer, train_texts, train_labels, train_pseudo_labels)
        valid_dataset = PhysicaliqaDatasetForKD(tokenizer, valid_texts, valid_labels, valid_pseudo_labels)
    print('training_size:', args.training_size)
    print('Train Set:', len(train_dataset), 'Valid Set:', len(valid_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch_KD(batch))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch_KD(batch))
    
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
        path = os.path.join(args.root_dir, 'models', 'STKD')
        condition = f'{args.lm}-kd-{args.pre_task}'
    else:
        path = os.path.join(args.root_dir, 'models', 'FT')
        condition = f'{args.lm}-{args.pre_task}'
    best_name, _ = get_best_model(path, condition)
    best_path = os.path.join(path, best_name)
    restore_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(restore_dict)

    # fine tuning LMs on QA datasets
    save_dir = os.path.join(args.root_dir, 'models', 'STKD')
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = CriterionForKD(args.alpha, args.T)
    trainer = TrainerForKD(model, optimizer, criterion, device)
    trainer.run_training(args, train_loader, valid_loader, target_names, save_dir)


if __name__ == '__main__':
    args = parser_args()
    main(args)