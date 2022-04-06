import os
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, AdamW, RobertaModel

from sklearn.metrics import classification_report, accuracy_score

from dataset import load_piqa, load_siqa, load_csqa, load_cmqa, load_piqa
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset
from model import Multiple_Choice_Model
from utils import get_best_model, test


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur_dir', type=str, default='/home/chaehyeong/CKL')
    parser.add_argument('--root_dir', type=str, default='/home/chaehyeong/nas')

    parser.add_argument('--lm', type=str, required=True, choices=['roberta-large', 'roberta-cskg'], help='Pre-trained LM or KG fine-tuned LM')
    parser.add_argument('--pre_task', type=str, default=None)
    parser.add_argument('--cur_task', type=str, required=True, choices=['siqa', 'csqa', 'cmqa', 'piqa'])
    parser.add_argument('--training_size', type=float, required=True, help='Training data size for fine-tuning LM')
    parser.add_argument('--target_task', type=str, required=True, choices=['siqa', 'csqa', 'cmqa', 'piqa'], help='Which QA dataset to use for evaluating LM')

    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    return args


def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset and dataloader
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    if args.target_task == 'siqa':
        target_names = ['Answer A', 'Answer B', 'Answer C']
        test_texts, test_labels = load_siqa(args.root_dir, 'dev')
        test_dataset = SocialiqaDataset(tokenizer, test_texts, test_labels)
    elif args.target_task == 'csqa':
        target_names = ['Answer A', 'Answer B', 'Answer C', 'Answer D', 'Answer E']
        test_texts, test_labels = load_csqa(args.root_dir, 'dev')
        test_dataset = CommonsenseqaDataset(tokenizer, test_texts, test_labels)
    elif args.target_task == 'cmqa':
        target_names = ['Answer A', 'Answer B', 'Answer C', 'Answer D']
        test_texts, test_labels = load_cmqa(args.root_dir, 'dev')
        test_dataset = CosmosqaDataset(tokenizer, test_texts, test_labels)
    elif args.target_task == 'piqa':
        target_names = ['Answer A', 'Answer B']
        test_texts, test_labels = load_piqa(args.root_dir, 'dev')
        test_dataset = PhysicaliqaDataset(tokenizer, test_texts, test_labels)
    print('Test Set:', len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # define model
    if args.lm == 'roberta-large':
        model_name = args.lm
    elif args.lm == 'roberta-cskg':
        model_name = os.path.join(args.root_dir, args.lm)
    roberta_model = RobertaModel.from_pretrained(model_name)
    model = Multiple_Choice_Model(roberta_model)

    # load fine-tuned model with the best performance
    path = os.path.join(args.root_dir, 'models', 'FT')
    condition = f'{args.lm}-{args.cur_task}-ts{args.training_size}-bs{args.batch_size}'
    best_name, best_acc = get_best_model(path, condition)
    best_path = os.path.join(path, best_name)
    restore_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(restore_dict)
    print('Best model:', best_name)

    # evaluate fine-tuned model
    _, labels, preds = test(model, test_loader, device)
    print(classification_report(labels, preds, target_names=target_names))

    # write evaluation results
    with open(os.path.join(args.root_dir, 'results', args.target_task, f'{best_name[:-2]}txt'), 'w') as f:
        f.write(f'{accuracy_score(labels, preds)}\n')
        f.write(classification_report(labels, preds, target_names=target_names))


if __name__ == '__main__':
    args = parser_args()
    main(args)
