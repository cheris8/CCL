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
from utils import make_dir, get_best_model_path, get_predictions_from_model, get_answer_options_pool, right_wrong_split, modify_wrong_to_right_samples, sort_samples_by_similarity


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def main(args):
    
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
        restore_dict = torch.load(best_path, map_location=args.device)
        model.load_state_dict(restore_dict)

    # dataset and dataloader
    target_names = TASK_CONFIG[args.cur_task]['target_names']
    num_choices = TASK_CONFIG[args.cur_task]['num_choices']
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    texts, labels = TASK_CONFIG[args.cur_task]['load'](args.root_dir, 'train')
    answer_options_pool = get_answer_options_pool(texts, num_choices)

    if 'Sampling' in args.training_type:
        logits = get_predictions_from_model(args, model, tokenizer, texts, labels)
        train_texts, valid_texts, train_labels, valid_labels, train_logits, valid_logits = train_test_split(texts, labels, logits, test_size=0.1, random_state=42)
        train_texts, train_labels, train_logits = sort_samples_by_similarity(args, train_texts, train_labels, train_logits, args.sampling_type)
        train_texts = train_texts[:args.training_size] ; train_labels = train_labels[:args.training_size] ; train_logits = train_logits[:args.training_size] 
        if 'KD' in args.training_type:
            train_dataset = TASK_CONFIG[args.cur_task]['dataset-kd'](tokenizer, train_texts, train_labels, train_logits)
            valid_dataset = TASK_CONFIG[args.cur_task]['dataset-kd'](tokenizer, valid_texts, valid_labels, valid_logits)
        else:
            train_dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, train_texts, train_labels)
            valid_dataset = TASK_CONFIG[args.cur_task]['dataset'](tokenizer, valid_texts, valid_labels)
    elif 'KD' in args.training_type:
        logits = get_predictions_from_model(args, model, tokenizer, texts, labels)
        train_texts, valid_texts, train_labels, valid_labels, train_logits, valid_logits = train_test_split(texts, labels, logits, train_size=args.training_size, random_state=42)
        if 'STDKD' in args.training_type:
            # only gather incorrect samples
            _, train_texts, _, train_labels, _, train_logits = right_wrong_split(train_texts, train_labels, train_logits)
        elif 'STRKD' in args.training_type:
            # replace incorrect samples to correct samples
            train_texts, wrong_texts, train_labels, wrong_labels, train_logits, _ = right_wrong_split(train_texts, train_labels, train_logits)
            additional_train_texts, additional_train_labels, additional_train_logits = modify_wrong_to_right_samples(args, model, tokenizer, wrong_texts, wrong_labels, answer_options_pool)
            train_texts.extend(additional_train_texts)
            train_labels.extend(additional_train_labels)
            train_logits.extend(additional_train_logits)
        # elif 'STSKD':
        elif 'STAKD' in args.training_type:
            # augment correctly modified samples
            if os.path.exists(os.path.join(args.base_dir, 'additional_samples.json')):
                print('Load saved additional samples ...')
                with open(os.path.join(args.base_dir, 'additional_samples.json'), 'r') as f:
                    json_file = json.load(f)
                    additional_train_texts = json_file['texts']
                    additional_train_labels = json_file['labels']
                    additional_train_logits = json_file['logits']
            else:
                print('Generate additional samples ...')
                _, wrong_texts, _, wrong_labels, _, _ = right_wrong_split(train_texts, train_labels, train_logits)
                additional_train_texts, additional_train_labels, additional_train_logits = modify_wrong_to_right_samples(args, model, tokenizer, wrong_texts, wrong_labels, answer_options_pool)
            train_texts.extend(additional_train_texts)
            train_labels.extend(additional_train_labels)
            train_logits.extend(additional_train_logits)
        
        train_dataset = TASK_CONFIG[args.cur_task]['dataset-kd'](tokenizer, train_texts, train_labels, train_logits)
        valid_dataset = TASK_CONFIG[args.cur_task]['dataset-kd'](tokenizer, valid_texts, valid_labels, valid_logits)
    else:
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(texts, labels, train_size=args.training_size, random_state=42)
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
        trainer = TRAINING_CONFIG[args.training_type]['trainer'](model, optimizer, criterion, args.device)
    else:
        trainer = TRAINING_CONFIG[args.training_type]['trainer'](model, optimizer, args.device)
    trainer.run_training(args, train_loader, valid_loader, target_names)


if __name__ == '__main__':
    main(args)