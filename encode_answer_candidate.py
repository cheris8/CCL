from utils import get_answer_options_pool
import os
import torch
from transformers import RobertaTokenizer, RobertaModel, AdamW
from model import BiEncoder
from torch.utils.data import DataLoader
from dataset import singleencoder_batch, SingleEncoderDataset
from tqdm.auto import tqdm
from utils import make_dir, TrainerForEncoder
import json
import pickle
import argparse
from dataset import load_siqa, load_csqa, load_cmqa, load_piqa, prepare_batch_KD, prepare_batch
from dataset import SocialiqaDataset, CommonsenseqaDataset, CosmosqaDataset, PhysicaliqaDataset
from dataset import SocialiqaDatasetForKD, CommonsenseqaDatasetForKD, CosmosqaDatasetForKD, PhysicaliqaDatasetForKD
from utils import CriterionForKD, TrainerForKD, Trainer, CriterionForSKD


TASK_CONFIG = {
    'siqa':{'load':load_siqa, 'dataset':SocialiqaDataset, 'dataset-kd':SocialiqaDatasetForKD, 'context':True, 'question':True, 'num_choices':3, 'target_names':['Answer A', 'Answer B', 'Answer C'] },
    'csqa':{'load':load_csqa, 'dataset':CommonsenseqaDataset, 'dataset-kd':CommonsenseqaDatasetForKD, 'context':False, 'question':True, 'num_choices':5, 'target_names':['Answer A', 'Answer B', 'Answer C', 'Answer D', 'Answer E']},
    'cmqa':{'load':load_cmqa, 'dataset':CosmosqaDataset, 'dataset-kd':CosmosqaDatasetForKD, 'context':True, 'question':True, 'num_choices':4, 'target_names':['Answer A', 'Answer B', 'Answer C', 'Answer D']},
    'piqa':{'load':load_piqa, 'dataset':PhysicaliqaDataset, 'dataset-kd':PhysicaliqaDatasetForKD, 'context':False, 'question':True, 'num_choices':2, 'target_names':['Answer A', 'Answer B']}
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_size', type=int, default=9000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--n_epoch', type=int, default=20)
    args = parser.parse_args()
    return args

def get_best_loss_path(path):
    all_files = os.listdir(path)
    files = [file for file in all_files if file.endswith('.pt')]
    losses = [float(file[-9:-3]) for file in files]
    assert len(files) == len(losses)
    loss_to_file = {}
    for file, loss in zip(files, losses):
        loss_to_file[loss] = file
    best_loss = min(loss_to_file.keys())
    return os.path.join(path, loss_to_file.get(best_loss))


def main(args):
    tasks = ['csqa', 'cmqa', 'siqa'] ## piqa not included!
    root_dir ='/home/intern/nas/CCL/bi-encoder'
    roberta_model_1 = RobertaModel.from_pretrained('roberta-large')
    roberta_model_2 = RobertaModel.from_pretrained('roberta-large')
    model = BiEncoder(roberta_model_1, roberta_model_2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    for task in tasks:
        path=os.path.join(root_dir,f'{task}-ts{args.training_size}')
        num_choices = TASK_CONFIG[task]['num_choices']
        texts, labels = TASK_CONFIG[task]['load']('/home/intern/nas', 'train')
        answer_options_pool = get_answer_options_pool(texts, num_choices)

        best_path = get_best_loss_path(path)
        print(f"best path: {best_path}")
        restore_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(restore_dict)
        model = model.to(device)
        model.eval()
        answer_dict={}
        for answer in answer_options_pool:
            train_dataset = SingleEncoderDataset(tokenizer, answer)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: singleencoder_batch(batch))
            with torch.no_grad():
                for batch in tqdm(train_loader):
                    input_ids, attention_mask =batch
                    result = model.encode_response(input_ids=input_ids.to(device), input_masks=attention_mask.to(device)).detach().cpu()
                    answer_dict[answer]=result
                    torch.cuda.empty_cache()
        base_dir ='/home/intern/nas/CCL/encoder'
        make_dir(base_dir)
        file_name = f'{task}-dict.pickle'
        file_dir = os.path.join(base_dir, file_name)
        with open(file_dir, 'wb') as wp:
            pickle.dump(answer_dict, wp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parser_args()
    main(args)