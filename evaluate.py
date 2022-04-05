import os
import json
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, AdamW, RobertaModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from settings import args, TASK_CONFIG, TRAINING_CONFIG
from model import Multiple_Choice_Model
from utils import get_best_model_path, test


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define model
    if args.lm == 'roberta-large':
        model_name = args.lm
    elif args.lm == 'roberta-cskg':
        model_name = os.path.join(args.root_dir, args.lm)
    roberta_model = RobertaModel.from_pretrained(model_name)
    model = Multiple_Choice_Model(roberta_model)

    # load fine-tuned model with the best performance
    best_path = get_best_model_path(args.base_dir)
    restore_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(restore_dict)
    print('Best model:', args.base_dir, best_path)

    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    results = {}
    for target_task in ['siqa', 'csqa', 'cmqa', 'piqa']:
        # dataset and dataloader
        test_texts, test_labels = TASK_CONFIG[target_task]['load'](args.root_dir, 'dev')
        test_dataset = TASK_CONFIG[target_task]['dataset'](tokenizer, test_texts, test_labels)
        print('Task', target_task, 'Test Set:', len(test_dataset))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # evaluate fine-tuned model
        _, labels, preds = test(model, test_loader, device)
        metrics = {
            'accuracy':accuracy_score(labels, preds),
            'f1-macro':f1_score(labels, preds, average='macro'),
            'f1-micro':f1_score(labels, preds, average='micro'),
            'precision-macro':precision_score(labels, preds, average='macro'),
            'precision-micro':precision_score(labels, preds, average='micro'),
            'recall-macro':recall_score(labels, preds, average='macro'),
            'recall-micro':recall_score(labels, preds, average='micro')
        }
        results[target_task] = metrics

        target_names = TASK_CONFIG[target_task]['target_names']
        print(classification_report(labels, preds, target_names=target_names))

    with open(os.path.join(args.base_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    #args = parser_args()
    main(args)
