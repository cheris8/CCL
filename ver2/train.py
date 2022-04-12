import os
import json
import random
import pathlib
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, AdamW

from sklearn.model_selection import train_test_split

from settings import args, TASK_CONFIG, TRAINING_CONFIG
from model import Multiple_Choice_Model
from utils import make_dir, get_best_model_path, replace_wrong_to_right_samples, get_predictions_from_teacher_model


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    make_dir(args.base_dir)
    print('Directory:', args.base_dir)

    # define model
    if args.lm == 'roberta-large':
        model_name = args.lm
    elif args.lm == 'roberta-cskg':
        model_name = os.path.join(args.root_dir, args.lm)
    roberta_model = RobertaModel.from_pretrained(model_name)
    model = Multiple_Choice_Model(roberta_model)

    # load trained model
    if 'ST' in args.training_type:
        if len(args.pre_task) > 1:
            path = os.path.join(args.pre_task_dir, args.training_type)
        else:
            path = os.path.join(args.pre_task_dir, 'FT')
        best_path = get_best_model_path(path)
        restore_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(restore_dict)


    # load teacher model
    if 'KD' in args.training_type:
        teacher_model = Multiple_Choice_Model(roberta_model)
        teacher_model.load_state_dict(restore_dict)

    # dataset and dataloader
    target_names = TASK_CONFIG[args.cur_task]['target_names']
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    train_texts, train_labels = TASK_CONFIG[args.cur_task]['load'](args.root_dir, 'train')

    if 'KD' in args.training_type:
        train_teacher_logits = get_predictions_from_teacher_model(args, teacher_model, tokenizer, train_texts, train_labels, device)
        train_texts, valid_texts, train_labels, valid_labels, train_teacher_logits, valid_teacher_logits \
            = train_test_split(train_texts, train_labels, train_teacher_logits, train_size=args.training_size, random_state=42)
        if args.training_type == 'STAKD':
            if os.path.exists(os.path.join(args.base_dir, 'additional_samples.json')):
                print('Load saved additional samples ...')
                with open(os.path.join(args.base_dir, 'additional_samples.json'), 'r') as f:
                    json_file = json.load(f)
                    additional_train_texts = json_file['texts']
                    additional_train_labels = json_file['labels']
                    additional_train_teacher_logits = json_file['logits']
            else:
                print('Generate additional samples ...')
                additional_train_texts, additional_train_labels, additional_train_teacher_logits = replace_wrong_to_right_samples(args, teacher_model, tokenizer, train_texts, train_labels, device)
            train_texts.extend(additional_train_texts)
            train_labels.extend(additional_train_labels)
            train_teacher_logits.extend(additional_train_teacher_logits)
        train_dataset = TASK_CONFIG[args.cur_task]['dataset-kd'](tokenizer, train_texts, train_labels, train_teacher_logits)
        valid_dataset = TASK_CONFIG[args.cur_task]['dataset-kd'](tokenizer, valid_texts, valid_labels, valid_teacher_logits)
    else:
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, train_size=args.training_size, random_state=42)
        train_dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, train_texts, train_labels)
        valid_dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, valid_texts, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: TRAINING_CONFIG[args.training_type]['collate_fn'](batch))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: TRAINING_CONFIG[args.training_type]['collate_fn'](batch))

    print('Tasks:', args.pre_task, args.cur_task)
    print('training_size:', args.training_size)
    print('Train Set:', len(train_dataset), 'Valid Set:', len(valid_dataset))
    

    # train LMs on QA datasets
    optimizer = AdamW(model.parameters(), lr=args.lr)
    if 'KD' in args.training_type:
        criterion = TRAINING_CONFIG[args.training_type]['criterion'](args.alpha, args.T)
        trainer = TRAINING_CONFIG[args.training_type]['trainer'](model, optimizer, criterion, device)
    else:
        trainer = TRAINING_CONFIG[args.training_type]['trainer'](model, optimizer, device)
    trainer.run_training(args, train_loader, valid_loader, target_names)


if __name__ == '__main__':
    main(args)