import os
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, RobertaModel, AdamW

from sklearn.model_selection import train_test_split

from dataset import load_siqa, load_csqa, load_cmqa, load_piqa, prepare_batch_KD, prepare_batch
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset
from dataset import SocialiqaDatasetForKD, CommonsenseqaDatasetForKD, CosmosqaDatasetForKD, PhysicaliqaDatasetForKD
from utils import CriterionForKD, TrainerForKD, Trainer, CriterionForSKD


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur_dir', type=str, default='/home/intern/seungjun/commonsense/CCL')
    parser.add_argument('--root_dir', type=str, default='/home/intern/nas')

    parser.add_argument('--lm', type=str, default='roberta-large', choices=['roberta-large', 'roberta-cskg'], help='Pre-trained LM or KG fine-tuned LM.')
    parser.add_argument('--pre_task', type=str, nargs='+', help='Which QA dataset is used for trainining LM.')
    parser.add_argument('--cur_task', type=str, required=True, choices=['siqa', 'csqa', 'cmqa', 'piqa'], help='Which QA dataset to use for training LM.')

    parser.add_argument('--training_type', choices=['FT', 'ST', 'STKD', 'STDKD', 'STSKD', 'STAKD', 'STRKD','SamplingRandomST', 'SamplingUniformST', 'SamplingSkewedST', 'SamplingSkewedSTKD', 'SamplingReversedST', 'SamplingReversedSTKD'])
    parser.add_argument('--split_type', choices=['prob', 'num'])
    parser.add_argument('--training_size', required=True, help='Number/Proportion of samples to use for training LM.')
    parser.add_argument('--sampling_type', choices=['random', 'uniform', 'skewed', 'reversed'])
    parser.add_argument('--negative', type=str, choices=['sbert', 'biencoder'], default=None)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--T', type=float, default=1)
    args = parser.parse_args()

    if args.split_type == 'prob':
        args.training_size = float(args.training_size)
    elif args.split_type == 'num':
        args.training_size = int(args.training_size)
    
    if 'ST' in args.training_type:
        args.pre_task_dir = os.path.join(
            args.root_dir, 'CCL', args.lm, 
            f'-ts{args.training_size}-'.join(args.pre_task)+f'-ts{args.training_size}'
        )
    
    if args.negative is not None:
        args.base_dir = os.path.join(
            args.root_dir, 'CCL', args.lm, 
            f'-ts{args.training_size}-'.join(args.pre_task) + f'-ts{args.training_size}-' + args.cur_task + f'-ts{args.training_size}' if 'ST' in args.training_type else args.cur_task + f'-ts{args.training_size}',
            args.training_type, args.negative
            )
    else:
        args.base_dir = os.path.join(
            args.root_dir, 'CCL', args.lm, 
            f'-ts{args.training_size}-'.join(args.pre_task) + f'-ts{args.training_size}-' + args.cur_task + f'-ts{args.training_size}' if 'ST' in args.training_type else args.cur_task + f'-ts{args.training_size}',
            args.training_type
            )

    if args.negative == 'biencoder':
        args.pickle_dir = os.path.join(
            args.root_dir, 'CCL', 'encoder', f'{args.cur_task}-dict.pickle'
        )
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return args

args = parser_args()

TASK_CONFIG = {
    'siqa':{'load':load_siqa, 'dataset':SocialiqaDataset, 'dataset-kd':SocialiqaDatasetForKD, 'context':True, 'question':True, 'num_choices':3, 'target_names':['Answer A', 'Answer B', 'Answer C'] },
    'csqa':{'load':load_csqa, 'dataset':CommonsenseqaDataset, 'dataset-kd':CommonsenseqaDatasetForKD, 'context':False, 'question':True, 'num_choices':5, 'target_names':['Answer A', 'Answer B', 'Answer C', 'Answer D', 'Answer E']},
    'cmqa':{'load':load_cmqa, 'dataset':CosmosqaDataset, 'dataset-kd':CosmosqaDatasetForKD, 'context':True, 'question':True, 'num_choices':4, 'target_names':['Answer A', 'Answer B', 'Answer C', 'Answer D']},
    'piqa':{'load':load_piqa, 'dataset':PhysicaliqaDataset, 'dataset-kd':PhysicaliqaDatasetForKD, 'context':False, 'question':True, 'num_choices':2, 'target_names':['Answer A', 'Answer B']}
}

TRAINING_CONFIG = {
    'FT': {'trainer':Trainer, 'collate_fn':prepare_batch},
    'ST': {'trainer':Trainer, 'collate_fn':prepare_batch},
    'STKD': {'criterion':CriterionForKD, 'trainer':TrainerForKD, 'collate_fn':prepare_batch_KD},
    'STDKD': {'criterion':CriterionForKD, 'trainer':TrainerForKD, 'collate_fn':prepare_batch_KD},
    'STRKD': {'criterion':CriterionForKD, 'trainer':TrainerForKD, 'collate_fn':prepare_batch_KD},
    'STSKD': {'criterion':CriterionForSKD, 'trainer':TrainerForKD, 'collate_fn':prepare_batch_KD},
    'STAKD': {'criterion':CriterionForSKD, 'trainer':TrainerForKD, 'collate_fn':prepare_batch_KD},
    'SamplingRandomST': {'trainer':Trainer, 'collate_fn':prepare_batch},
    'SamplingUniformST': {'trainer':Trainer, 'collate_fn':prepare_batch},
    'SamplingSkewedST': {'trainer':Trainer, 'collate_fn':prepare_batch},
    'SamplingSkewedSTKD': {'criterion':CriterionForKD, 'trainer':TrainerForKD, 'collate_fn':prepare_batch_KD},
    'SamplingReversedST': {'trainer':Trainer, 'collate_fn':prepare_batch},
    'SamplingReversedSTKD': {'criterion':CriterionForKD, 'trainer':TrainerForKD, 'collate_fn':prepare_batch_KD}
}